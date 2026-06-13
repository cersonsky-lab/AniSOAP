#!/usr/bin/env python3
"""
Finite-difference and autograd checks for SOAP-BPNN.

This is the SOAP analogue of anisoap_bpnn_fd_checks.py.

It does three things:

1. Trains a SOAP-BPNN energy model with energy + position-gradient targets.
2. Checks energy parity with the exported model.
3. Checks forces in two ways:
   - autograd forces from the non-exported PyTorch model
   - finite-difference forces from the exported model

Important distinction:
    exported(...) + energy.backward() is not expected to work.
    Use the non-exported model for manual autograd.
    Use the exported model for finite differences / deployment-style evaluation.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read
from metatomic.torch import ModelEvaluationOptions, ModelOutput, systems_to_torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from omegaconf import OmegaConf
from torch.utils.data import random_split

from metatrain.soap_bpnn import SoapBpnn as BPNN
from metatrain.soap_bpnn import Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


def build_hypers(lmax: int = 1, nmax: int = 1) -> dict:
    """Build SOAP-BPNN hypers."""
    return {
        "soap": {
            "cutoff": {
                "radius": 4.5,
                "smoothing": {"type": "ShiftedCosine", "width": 0.5},
                "width": 0.5,
            },
            "density": {
                "type": "Gaussian",
                "width": 1.0,
            },
            "basis": {
                "type": "TensorProduct",
                "max_angular": lmax,
                "radial": {"type": "Gto", "max_radial": nmax},
            },
            "max_angular": lmax,
            "max_radial": nmax,
        },
        "bpnn": {
            "num_hidden_layers": 2,
            "num_neurons_per_layer": 32,
            "layernorm": True,
        },
        "legacy": False,
        "long_range": {"enable": False},
        "heads": {},
        "add_lambda_basis": False,
        "zbl": False,
    }


def build_dataset_info() -> DatasetInfo:
    """Build DatasetInfo for a single pseudo-species system with energy+forces."""
    target_cfg = OmegaConf.create(
        {
            "quantity": "energy",
            "unit": "eV",
        }
    )

    energy_info = get_energy_target_info(
        target_name="energy",
        target=target_cfg,
        add_position_gradients=True,
    )

    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=[0],
        targets={"energy": energy_info},
    )


def energy_target(atoms, system_i: int) -> TensorMap:
    """
    Build an energy target TensorMap with a positions gradient.

    metatensor target gradients store dE/dr. ASE forces are -dE/dr, so the
    gradient values are -forces.
    """
    n_atoms = len(atoms)

    properties = Labels(
        ["energy"],
        torch.tensor([[0]], dtype=torch.int32),
    )

    block = TensorBlock(
        values=torch.tensor([[atoms.get_potential_energy()]], dtype=torch.float64),
        samples=Labels(
            ["system"],
            torch.tensor([[system_i]], dtype=torch.int32),
        ),
        components=[],
        properties=properties,
    )

    forces = torch.tensor(atoms.get_forces(), dtype=torch.float64)

    grad_block = TensorBlock(
        values=-forces.reshape(n_atoms, 3, 1),
        samples=Labels(
            ["sample", "atom"],
            torch.tensor([[0, i] for i in range(n_atoms)], dtype=torch.int32),
        ),
        components=[
            Labels(
                ["xyz"],
                torch.tensor([[0], [1], [2]], dtype=torch.int32),
            )
        ],
        properties=properties,
    )

    block.add_gradient("positions", grad_block)

    return TensorMap(keys=Labels.single(), blocks=[block])


def load_dataset(path: Path, stride: int = 18) -> tuple[list, list]:
    """Read ASE frames and convert them to metatomic torch Systems."""
    frames = read(path, ":")[::stride]

    # Keep original frames untouched so their SinglePointCalculator still has
    # energy/forces. Zero dummy non-periodic cells only on copies used for
    # System conversion.
    frames_for_systems = []
    for frame in frames:
        frame_for_system = frame.copy()

        if not frame_for_system.pbc.any():
            frame_for_system.cell = [0.0, 0.0, 0.0]

        frames_for_systems.append(frame_for_system)

    systems = systems_to_torch(frames_for_systems, dtype=torch.float64)
    for system in systems:
        system.positions.requires_grad_(True)

    return frames, systems


def build_dataset(frames: list, systems: list) -> Dataset:
    """Build a metatrain Dataset from systems and energy/force targets."""
    return Dataset.from_dict(
        {
            "system": systems,
            "energy": [energy_target(atoms, i) for i, atoms in enumerate(frames)],
        }
    )


def build_trainer(num_epochs: int, batch_size: int, learning_rate: float) -> Trainer:
    """Build the SOAP-BPNN trainer."""
    return Trainer(
        {
            "distributed": False,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "atomic_baseline": {"energy": {0: 0}},
            "scale_targets": True,
            "fixed_scaling_weights": {"energy": 1.0},
            "batch_atom_bounds": [None, None],
            "num_workers": 0,
            "loss": {
                "type": "mse",
                "weights": {
                    "energy": 1.0,
                    "forces": 10.0,
                },
            },
            "warmup_fraction": 0.1,
            "per_structure_targets": ["energy"],
            "per_atom_targets": ["forces"],
            "log_separate_blocks": True,
            "log_mae": True,
            "log_interval": 1,
            "checkpoint_interval": 1,
            "best_model_metric": "rmse_prod",
        }
    )


def energy_model_outputs() -> dict[str, ModelOutput]:
    """Outputs dictionary for calling the non-exported PyTorch model."""
    return {
        "energy": ModelOutput(
            quantity="energy",
            unit="eV",
            sample_kind="system",
        )
    }


def exported_energy_options() -> ModelEvaluationOptions:
    """Evaluation options for calling the exported metatomic model."""
    return ModelEvaluationOptions(
        length_unit="angstrom",
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
                sample_kind="system",
            )
        },
    )


def with_neighbor_lists(model: BPNN, systems: Iterable) -> list:
    """Attach the neighbor lists requested by the model."""
    requested_neighbor_lists = get_requested_neighbor_lists(model)
    return [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in systems
    ]


def true_energy(sample) -> float:
    return sample["energy"].block(0).values.item()


def true_forces(sample) -> torch.Tensor:
    """
    Return physical forces from the target TensorMap.

    The stored gradient is dE/dr = -force, so physical force = -gradient.
    """
    return -sample["energy"].block(0).gradient("positions").values.squeeze(-1)


def predict_energy_exported(exported, system) -> torch.Tensor:
    """Predict total energy with the exported model."""
    pred = exported(
        [system],
        exported_energy_options(),
        check_consistency=False,
    )
    return pred["energy"].block(0).values.sum()


def predict_energy_torch(model: BPNN, system) -> torch.Tensor:
    """Predict total energy with the non-exported torch model."""
    pred = model(
        [system],
        energy_model_outputs(),
    )
    return pred["energy"].block(0).values.sum()


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path | None = None,
) -> None:
    """Make a parity plot."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=20, alpha=0.7)

    lims = [
        min(float(y_true.min()), float(y_pred.min())),
        max(float(y_true.max()), float(y_pred.max())),
    ]

    plt.plot(lims, lims, "k--", lw=2)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.gca().set_aspect("equal")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nRMSE = {rmse:.3e}, MAE = {mae:.3e}")

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
    else:
        plt.show()

    plt.close()


def exported_energy_parity(exported, dataset: Dataset, output_dir: Path) -> None:
    """Check exported-model energy parity."""
    y_true = []
    y_pred = []

    for sample in dataset:
        system = sample["system"]
        pred = exported(
            [system],
            exported_energy_options(),
            check_consistency=True,
        )

        y_true.append(true_energy(sample))
        y_pred.append(pred["energy"].block(0).values.item())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plot_parity(
        y_true,
        y_pred,
        xlabel="True energy / eV",
        ylabel="Predicted energy / eV",
        title="Energy parity, exported SOAP-BPNN model",
        output_path=output_dir / "soap_bpnn_energy_parity_exported.png",
    )


def autograd_force_parity(model: BPNN, dataset: Dataset, output_dir: Path) -> None:
    """
    Compute forces by differentiating the non-exported torch model.

    Do not use exported(...).backward() for this test.
    """
    f_true = []
    f_pred = []

    was_training = model.training
    model.train()

    try:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("trainable parameters:", n_trainable)

        for sample_i, sample in enumerate(dataset):
            system = copy.deepcopy(sample["system"])
            system.positions.requires_grad_(True)

            system = with_neighbor_lists(model, [system])[0]

            model.zero_grad(set_to_none=True)
            if system.positions.grad is not None:
                system.positions.grad.zero_()

            energy = predict_energy_torch(model, system)

            print(
                f"system {sample_i}: "
                f"energy requires_grad={energy.requires_grad}, grad_fn={energy.grad_fn}"
            )

            forces_true = true_forces(sample).to(system.positions)

            if not energy.requires_grad:
                # If no contributing features exist for a structure, the model
                # energy can be constant w.r.t. positions, giving exactly zero
                # model forces.
                forces_pred = torch.zeros_like(forces_true)
            else:
                energy.backward()

                if system.positions.grad is None:
                    forces_pred = torch.zeros_like(forces_true)
                else:
                    forces_pred = -system.positions.grad.detach()

            f_pred.append(forces_pred.cpu().numpy().ravel())
            f_true.append(forces_true.cpu().numpy().ravel())

    finally:
        if not was_training:
            model.eval()

    f_true = np.concatenate(f_true)
    f_pred = np.concatenate(f_pred)

    plot_parity(
        f_true,
        f_pred,
        xlabel="True force / eV Å$^{-1}$",
        ylabel="Autograd force / eV Å$^{-1}$",
        title="Force parity, non-exported SOAP-BPNN autograd",
        output_path=output_dir / "soap_bpnn_force_parity_autograd.png",
    )


def finite_difference_force_component(
    exported,
    system,
    atom_i: int,
    xyz: int,
    delta: float,
) -> torch.Tensor:
    """Central finite-difference force component from exported energies."""
    plus = copy.deepcopy(system)
    minus = copy.deepcopy(system)

    plus.positions[atom_i, xyz] += delta
    minus.positions[atom_i, xyz] -= delta

    e_plus = predict_energy_exported(exported, plus).detach()
    e_minus = predict_energy_exported(exported, minus).detach()

    return -(e_plus - e_minus) / (2.0 * delta)


def finite_difference_force_parity(
    exported,
    dataset: Dataset,
    output_dir: Path,
    delta: float,
    max_systems: int | None = None,
) -> None:
    """
    Compute finite-difference forces from exported model energies.

    This is deployment-style force testing: no autograd is used.
    """
    f_true = []
    f_pred = []

    samples = list(dataset)
    if max_systems is not None:
        samples = samples[:max_systems]

    for sample_i, sample in enumerate(samples):
        system = copy.deepcopy(sample["system"])
        forces_true = true_forces(sample)

        forces_fd = torch.zeros_like(forces_true)

        for atom_i in range(system.positions.shape[0]):
            for xyz in range(3):
                forces_fd[atom_i, xyz] = finite_difference_force_component(
                    exported=exported,
                    system=system,
                    atom_i=atom_i,
                    xyz=xyz,
                    delta=delta,
                )

        f_true.append(forces_true.cpu().numpy().ravel())
        f_pred.append(forces_fd.cpu().numpy().ravel())

        print(
            f"finite-difference forces: completed system {sample_i + 1}/{len(samples)}"
        )

    f_true = np.concatenate(f_true)
    f_pred = np.concatenate(f_pred)

    plot_parity(
        f_true,
        f_pred,
        xlabel="True force / eV Å$^{-1}$",
        ylabel="Finite-difference force / eV Å$^{-1}$",
        title=f"Force parity, exported SOAP-BPNN finite difference, delta={delta:g}",
        output_path=output_dir
        / "soap_bpnn_force_parity_finite_difference_exported.png",
    )


def train_model(
    model: BPNN,
    dataset: Dataset,
    output_dir: Path,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Train the model."""
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )

    trainer = build_trainer(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.train(
        model=model,
        dtype=torch.float64,
        devices=[torch.device("cpu")],
        train_datasets=[train_dataset],
        val_datasets=[val_dataset],
        checkpoint_dir=str(output_dir),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SOAP-BPNN and run finite-difference/autograd checks."
    )
    parser.add_argument(
        "--xyz",
        type=Path,
        default=Path("./all_frames.xyz"),
        help="Input ASE-readable XYZ/extxyz file.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth frame from the input file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./soap_bpnn_model_outputs"),
        help="Directory for checkpoints and plots.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Training learning rate.",
    )
    parser.add_argument(
        "--fd-delta",
        type=float,
        default=1e-4,
        help="Finite difference displacement.",
    )
    parser.add_argument(
        "--fd-max-systems",
        type=int,
        default=-1,
        help=(
            "Maximum number of systems for finite-difference force parity. "
            "Use -1 for all systems."
        ),
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and run checks with the initialized model.",
    )
    parser.add_argument(
        "--skip-autograd",
        action="store_true",
        help="Skip non-exported autograd force parity.",
    )
    parser.add_argument(
        "--skip-fd",
        action="store_true",
        help="Skip exported finite-difference force parity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    dataset_info = build_dataset_info()
    hypers = build_hypers(lmax=6, nmax=9)

    model = BPNN(hypers=hypers, dataset_info=dataset_info)

    print("target info:", dataset_info.targets["energy"])
    print("reading:", args.xyz)

    frames, systems = load_dataset(args.xyz, stride=args.stride)
    dataset = build_dataset(frames, systems)

    print(f"loaded {len(dataset)} systems")

    if not args.skip_train:
        train_model(
            model=model,
            dataset=dataset,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    exported = model.export()

    exported_energy_parity(
        exported=exported,
        dataset=dataset,
        output_dir=args.output_dir,
    )

    if not args.skip_autograd:
        autograd_force_parity(
            model=model,
            dataset=dataset,
            output_dir=args.output_dir,
        )

    if not args.skip_fd:
        max_systems = None if args.fd_max_systems < 0 else args.fd_max_systems
        finite_difference_force_parity(
            exported=exported,
            dataset=dataset,
            output_dir=args.output_dir,
            delta=args.fd_delta,
            max_systems=max_systems,
        )

    print("done")
    print(f"plots written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
