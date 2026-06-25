from __future__ import annotations

import argparse
import copy
import logging
import time
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

from metatrain.anisoap_bpnn import AnisoapBPNN as BPNN
from metatrain.anisoap_bpnn import Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

ANISOAP_QUATERNIONS = "anisoap::quaternions"
ANISOAP_C_DIAMETERS = "anisoap::c_diameters"
ENERGY_TARGET = "energy"


# -----------------------------------------------------------------------------
# TensorMap/System helpers
# -----------------------------------------------------------------------------


def per_atom_tensormap(
    array,
    *,
    property_name: str,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> TensorMap:
    """Convert a per-atom 1D/2D array to a single-block TensorMap."""
    values = torch.as_tensor(array, dtype=dtype, device=device)

    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(
            f"expected a 1D or 2D per-atom array, got shape {tuple(values.shape)}"
        )

    n_atoms, n_properties = values.shape
    block = TensorBlock(
        values=values,
        samples=Labels(
            ["atom"],
            torch.arange(n_atoms, dtype=torch.int32, device=device).reshape(-1, 1),
        ),
        components=[],
        properties=Labels(
            [property_name],
            torch.arange(n_properties, dtype=torch.int32, device=device).reshape(-1, 1),
        ),
    )
    return TensorMap(keys=Labels.single(), blocks=[block])


def replace_system_data(system, name: str, values: torch.Tensor):
    """
    Replace one single-block System data TensorMap while preserving labels.

    metatomic.torch.System.add_data(...) does not overwrite existing custom data.
    Since these diagnostics always operate on deep-copied Systems, mutate the
    existing TensorBlock values in place instead.
    """
    data_map = system.get_data(name)
    block = data_map.block(0)

    values = values.to(device=block.values.device, dtype=block.values.dtype)

    if tuple(values.shape) != tuple(block.values.shape):
        raise ValueError(
            f"replacement for {name!r} has shape {tuple(values.shape)}, "
            f"expected {tuple(block.values.shape)}"
        )

    block.values[:] = values
    return system


def replace_system_quaternions(system, quaternions: torch.Tensor):
    """Replace per-atom AniSOAP quaternions on a System."""
    return replace_system_data(system, ANISOAP_QUATERNIONS, quaternions)


# -----------------------------------------------------------------------------
# Quaternion/rotation helpers
# -----------------------------------------------------------------------------


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion product using [w, x, y, z] convention."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def small_rotation_quaternion(
    axis: int,
    angle: float,
    *,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """Quaternion for an active small rotation about x/y/z."""
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")

    half = torch.as_tensor(0.5 * angle, dtype=dtype, device=device)
    q = torch.zeros(4, dtype=dtype, device=device)
    q[0] = torch.cos(half)
    q[axis + 1] = torch.sin(half)
    return q


def normalized_quaternions(q: torch.Tensor) -> torch.Tensor:
    """Normalize quaternions row-wise."""
    return q / q.norm(dim=1, keepdim=True).clamp_min(1e-15)


# -----------------------------------------------------------------------------
# Model/data construction
# -----------------------------------------------------------------------------


def build_hypers(
    lmax: int = 3,
    nmax: int = 4,
    cutoff: float = 5.0,
    width: float = 0.5,
) -> dict:
    """Build AniSOAP-BPNN hypers in the format expected by this architecture."""
    hypers = {
        "soap": {
            "cutoff": {
                "radius": cutoff,
                "smoothing": {"type": "ShiftedCosine", "width": width},
            },
            "density": {"type": "Gaussian", "width": 1.0},
            "basis": {
                "type": "TensorProduct",
                "max_angular": lmax,
                "radial": {"type": "Gto", "max_radial": nmax},
            },
        },
        "krr": {"num_sparse_points": 10, "degree": 1},
        "zbl": False,
        "legacy": False,
        "long_range": {"enable": False},
        "heads": "mlp",
        "add_lambda_basis": False,
        "bpnn": {
            "num_hidden_layers": 2,
            "num_neurons_per_layer": 32,
            "layernorm": True,
        },
    }

    # Compatibility aliases used by the AniSOAP adapter path.
    hypers["soap"]["max_angular"] = hypers["soap"]["basis"]["max_angular"]
    hypers["soap"]["max_radial"] = hypers["soap"]["basis"]["radial"]["max_radial"]
    hypers["soap"]["cutoff"]["width"] = hypers["soap"]["cutoff"]["smoothing"]["width"]
    return hypers


def build_dataset_info() -> DatasetInfo:
    """Build DatasetInfo for one pseudo-species system with energy+forces."""
    target_cfg = OmegaConf.create({"quantity": "energy", "unit": "eV"})
    energy_info = get_energy_target_info(
        target_name=ENERGY_TARGET,
        target=target_cfg,
        add_position_gradients=True,
    )
    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=[0],
        targets={ENERGY_TARGET: energy_info},
    )


def energy_target(atoms, system_i: int) -> TensorMap:
    """Build energy target with position gradient dE/dr = -forces."""
    n_atoms = len(atoms)
    properties = Labels(["energy"], torch.tensor([[0]], dtype=torch.int32))

    block = TensorBlock(
        values=torch.tensor([[atoms.get_potential_energy()]], dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[system_i]], dtype=torch.int32)),
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
        components=[Labels(["xyz"], torch.tensor([[0], [1], [2]], dtype=torch.int32))],
        properties=properties,
    )
    block.add_gradient("positions", grad_block)
    return TensorMap(keys=Labels.single(), blocks=[block])


def torque_target(atoms, system_i: int) -> TensorMap:
    """Build a per-atom torque TensorMap from atoms.arrays['torques']."""
    if "torques" not in atoms.arrays:
        raise KeyError("ASE frame is missing atoms.arrays['torques']")

    torques = torch.as_tensor(atoms.arrays["torques"], dtype=torch.float64)
    if torques.ndim != 2 or torques.shape[1] != 3:
        raise ValueError(
            f"torques must have shape [n_atoms, 3], got {tuple(torques.shape)}"
        )

    n_atoms = torques.shape[0]
    block = TensorBlock(
        values=torques.reshape(n_atoms, 3, 1),
        samples=Labels(
            ["system", "atom"],
            torch.tensor([[system_i, i] for i in range(n_atoms)], dtype=torch.int32),
        ),
        components=[Labels(["xyz"], torch.tensor([[0], [1], [2]], dtype=torch.int32))],
        properties=Labels(["torque"], torch.tensor([[0]], dtype=torch.int32)),
    )
    return TensorMap(keys=Labels.single(), blocks=[block])


def load_dataset(path: Path, stride: int = 6) -> tuple[list, list]:
    """Read ASE frames and convert them to metatomic torch Systems."""
    frames = read(path, ":")[::stride]

    # Keep original frames untouched so SinglePointCalculator still has energy/forces.
    frames_for_systems = []
    for frame in frames:
        frame_for_system = frame.copy()
        if not frame_for_system.pbc.any():
            frame_for_system.cell = [0.0, 0.0, 0.0]
        frames_for_systems.append(frame_for_system)

    systems = systems_to_torch(frames_for_systems, dtype=torch.float64)

    for system, frame in zip(systems, frames):
        system.positions.requires_grad_(True)

        if "quaternions" not in frame.arrays:
            raise KeyError("ASE frame is missing frame.arrays['quaternions']")
        for name in ["c_diameter[1]", "c_diameter[2]", "c_diameter[3]"]:
            if name not in frame.arrays:
                print(frame.arrays)
                raise KeyError(f"ASE frame is missing frame.arrays[{name!r}]")

        q = np.asarray(frame.arrays["quaternions"], dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 4:
            raise ValueError(f"quaternions must have shape [n_atoms, 4], got {q.shape}")
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        if np.any(q_norm == 0.0):
            raise ValueError("found zero-norm quaternion")
        q = q / q_norm

        diameters = np.stack(
            [
                np.asarray(frame.arrays["c_diameter[1]"], dtype=np.float64),
                np.asarray(frame.arrays["c_diameter[2]"], dtype=np.float64),
                np.asarray(frame.arrays["c_diameter[3]"], dtype=np.float64),
            ],
            axis=1,
        )

        system.add_data(
            ANISOAP_QUATERNIONS,
            per_atom_tensormap(q, property_name="q", dtype=torch.float64),
        )
        system.add_data(
            ANISOAP_C_DIAMETERS,
            per_atom_tensormap(diameters, property_name="axis", dtype=torch.float64),
        )

    return frames, systems


def build_dataset(frames: list, systems: list) -> Dataset:
    """Build a metatrain Dataset from systems and energy/force/torque targets."""
    return Dataset.from_dict(
        {
            "system": systems,
            ENERGY_TARGET: [energy_target(atoms, i) for i, atoms in enumerate(frames)],
            "torques": [torque_target(atoms, i) for i, atoms in enumerate(frames)],
        }
    )


def build_trainer(
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    force_weight: float,
) -> Trainer:
    """Build the AniSOAP-BPNN trainer. Torques are evaluated, not trained here."""
    loss_weights = {ENERGY_TARGET: 1.0}
    per_atom_targets = []

    if force_weight > 0.0:
        loss_weights[f"{ENERGY_TARGET}_positions_gradients"] = force_weight
        per_atom_targets = ["forces"]
    return Trainer(
        {
            "regularizer": 1e-3,
            "regularizer_forces": 1e-3,
            "distributed": False,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "atomic_baseline": {},
            "scale_targets": False,
            "fixed_scaling_weights": {},
            "batch_atom_bounds": [None, None],
            "num_workers": 0,
            "loss": {
                "type": "mse",
                "weights": loss_weights,
            },
            "per_atom_targets": per_atom_targets,
            "warmup_fraction": 0.1,
            "per_structure_targets": [ENERGY_TARGET],
            "log_separate_blocks": True,
            "log_mae": True,
            "log_interval": 1,
            "checkpoint_interval": 1,
            "best_model_metric": "rmse_prod",
        }
    )


# -----------------------------------------------------------------------------
# Prediction/evaluation helpers
# -----------------------------------------------------------------------------


def energy_model_outputs() -> dict[str, ModelOutput]:
    """Outputs dictionary for calling the non-exported PyTorch model."""
    return {
        ENERGY_TARGET: ModelOutput(
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
            ENERGY_TARGET: ModelOutput(
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
    return sample[ENERGY_TARGET].block(0).values.item()


def true_forces(sample) -> torch.Tensor:
    """Return physical forces. Stored target gradient is dE/dr = -force."""
    return -sample[ENERGY_TARGET].block(0).gradient("positions").values.squeeze(-1)


def true_torques(sample) -> torch.Tensor:
    """Return physical torques from the torque target TensorMap."""
    return sample["torques"].block(0).values.squeeze(-1)


def predict_energy_torch(model: BPNN, system) -> torch.Tensor:
    """Predict total energy with the non-exported torch model."""
    pred = model([system], energy_model_outputs())
    return pred[ENERGY_TARGET].block(0).values.sum()


def predict_energy_exported(exported, system) -> torch.Tensor:
    """Predict total energy with the exported model."""
    pred = exported([system], exported_energy_options(), check_consistency=False)
    return pred[ENERGY_TARGET].block(0).values.sum()


def flattened_parameters(model: BPNN) -> torch.Tensor:
    """Return all model parameters flattened into one detached vector."""
    params = [p.detach().reshape(-1).cpu() for p in model.parameters()]
    return torch.cat(params) if params else torch.empty(0)


def print_parameter_change(
    before: torch.Tensor, after: torch.Tensor, label: str
) -> None:
    """Print max/mean absolute parameter change."""
    print(f"\n=== parameter change: {label} ===")
    if before.numel() == 0 and after.numel() == 0:
        print("no parameters found")
        return
    if before.shape != after.shape:
        print(
            "parameter vector shape changed:",
            tuple(before.shape),
            "->",
            tuple(after.shape),
        )
        return

    diff = (after - before).abs()
    print("num parameters:", diff.numel())
    print("max abs change:", diff.max().item())
    print("mean abs change:", diff.mean().item())
    print("num changed > 1e-14:", int((diff > 1e-14).sum().item()))


def print_progress(label: str, current: int, total: int, start_time: float) -> None:
    """Print simple elapsed-time progress for long loops."""
    elapsed = time.time() - start_time
    rate = current / elapsed if elapsed > 0.0 else 0.0
    remaining = (total - current) / rate if rate > 0.0 else float("nan")
    print(
        f"{label}: {current}/{total} elapsed={elapsed:.1f}s eta={remaining:.1f}s",
        flush=True,
    )


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    diff = y_pred - y_true
    return float(np.sqrt(np.mean(diff**2))), float(np.mean(np.abs(diff)))


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Make a parity plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rmse, mae = rmse_mae(y_true, y_pred)

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=20, alpha=0.7)

    low = min(float(y_true.min()), float(y_pred.min()))
    high = max(float(y_true.max()), float(y_pred.max()))
    if low == high:
        pad = 1.0 if low == 0.0 else abs(low) * 0.05
        low -= pad
        high += pad

    plt.plot([low, high], [low, high], "k--", lw=2)
    plt.xlim([low, high])
    plt.ylim([low, high])
    plt.gca().set_aspect("equal")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nRMSE = {rmse:.3e}, MAE = {mae:.3e}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


def print_ase_frame_diagnostics(frames: list, max_frames: int = 3) -> None:
    """Print basic diagnostics for the raw ASE frames."""
    print("\n=== ASE frame diagnostics ===")
    print("number of loaded frames:", len(frames))

    for frame_i, frame in enumerate(frames[:max_frames]):
        print(f"frame {frame_i}:")
        print("  symbols:", frame.get_chemical_symbols())
        print("  pbc:", frame.pbc)
        print("  cell lengths:", frame.cell.lengths())
        print("  arrays:", sorted(frame.arrays.keys()))
        print("  info keys:", sorted(frame.info.keys()))

        try:
            print("  energy:", frame.get_potential_energy())
        except Exception as exc:
            print("  energy read failed:", repr(exc))

        try:
            forces = frame.get_forces()
            print(
                "  force shape/norm/min/max:",
                forces.shape,
                float(np.linalg.norm(forces)),
                float(np.min(forces)),
                float(np.max(forces)),
            )
        except Exception as exc:
            print("  forces read failed:", repr(exc))

        for name in [
            "quaternions",
            "c_diameter[1]",
            "c_diameter[2]",
            "c_diameter[3]",
            "torques",
        ]:
            if name in frame.arrays:
                arr = frame.arrays[name]
                print(
                    f"  array {name}: shape={arr.shape}, first={arr[0] if len(arr) else None}"
                )


def print_model_diagnostics(model: BPNN) -> None:
    """Print basic model diagnostics."""
    print("\n=== model diagnostics ===")
    print("model class:", type(model))
    print(
        "trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    print("total parameters:", sum(p.numel() for p in model.parameters()))


def check_single_displacement_sensitivity(
    model: BPNN, sample, delta: float = 1e-3
) -> None:
    """Check whether one small displacement changes the non-exported model energy."""
    print("\n=== single displacement sensitivity ===")

    system0 = with_neighbor_lists(model, [copy.deepcopy(sample["system"])])[0]
    e0 = predict_energy_torch(model, system0).detach()

    system1 = copy.deepcopy(sample["system"])
    system1.positions[0, 0] += delta
    system1 = with_neighbor_lists(model, [system1])[0]
    e1 = predict_energy_torch(model, system1).detach()

    print("e0:", e0.item())
    print("e1:", e1.item())
    print("dE:", (e1 - e0).item())
    print("dE/dx approx:", ((e1 - e0) / delta).item())


def check_single_rotation_sensitivity(model: BPNN, sample, delta: float = 1e-3) -> None:
    """Check whether one small quaternion rotation changes the model energy."""
    print("\n=== single rotation sensitivity ===")

    system0 = with_neighbor_lists(model, [copy.deepcopy(sample["system"])])[0]
    e0 = predict_energy_torch(model, system0).detach()

    system1 = copy.deepcopy(sample["system"])
    q = system1.get_data(ANISOAP_QUATERNIONS).block(0).values.clone()
    dq = small_rotation_quaternion(2, delta, dtype=q.dtype, device=q.device)
    q[0] = quat_multiply(dq, q[0])
    q = normalized_quaternions(q)
    system1 = replace_system_quaternions(system1, q)
    system1 = with_neighbor_lists(model, [system1])[0]
    e1 = predict_energy_torch(model, system1).detach()

    print("e0:", e0.item())
    print("e1:", e1.item())
    print("dE:", (e1 - e0).item())
    print("dE/dtheta approx:", ((e1 - e0) / delta).item())


def torch_fd_force_component(
    model: BPNN, system, atom_i: int, xyz: int, delta: float
) -> torch.Tensor:
    """Central finite-difference force component from non-exported model energies."""
    plus = copy.deepcopy(system)
    minus = copy.deepcopy(system)
    plus.positions[atom_i, xyz] += delta
    minus.positions[atom_i, xyz] -= delta
    plus = with_neighbor_lists(model, [plus])[0]
    minus = with_neighbor_lists(model, [minus])[0]
    return -(
        predict_energy_torch(model, plus).detach()
        - predict_energy_torch(model, minus).detach()
    ) / (2.0 * delta)


def exported_fd_force_component(
    exported, model: BPNN, system, atom_i: int, xyz: int, delta: float
) -> torch.Tensor:
    """Central finite-difference force component from exported model energies."""
    plus = copy.deepcopy(system)
    minus = copy.deepcopy(system)
    plus.positions[atom_i, xyz] += delta
    minus.positions[atom_i, xyz] -= delta
    plus = with_neighbor_lists(model, [plus])[0]
    minus = with_neighbor_lists(model, [minus])[0]
    return -(
        predict_energy_exported(exported, plus).detach()
        - predict_energy_exported(exported, minus).detach()
    ) / (2.0 * delta)


def check_single_component_force_consistency(
    model: BPNN, sample, delta: float = 1e-4
) -> None:
    """Compare one force component by autograd, torch FD, exported FD, and target."""
    print("\n=== single-component force consistency ===")

    was_training = model.training
    model.train()
    try:
        system = copy.deepcopy(sample["system"])
        system.positions.requires_grad_(True)
        system = with_neighbor_lists(model, [system])[0]

        model.zero_grad(set_to_none=True)
        if system.positions.grad is not None:
            system.positions.grad.zero_()

        energy = predict_energy_torch(model, system)
        if energy.requires_grad:
            energy.backward()
            f_autograd = (
                torch.tensor(0.0, dtype=system.positions.dtype)
                if system.positions.grad is None
                else -system.positions.grad[0, 0].detach()
            )
        else:
            f_autograd = torch.tensor(0.0, dtype=system.positions.dtype)

        f_fd_torch = torch_fd_force_component(
            model, copy.deepcopy(sample["system"]), 0, 0, delta
        )
        exported = copy.deepcopy(model).export()
        f_fd_exported = exported_fd_force_component(
            exported, model, copy.deepcopy(sample["system"]), 0, 0, delta
        )
        f_true = true_forces(sample)[0, 0]

        print("component: atom=0, xyz=0")
        print("energy requires_grad:", energy.requires_grad)
        print("torch autograd force:", f_autograd.item())
        print("torch finite-difference force:", f_fd_torch.item())
        print("exported finite-difference force:", f_fd_exported.item())
        print("target force:", f_true.item())
        print("abs autograd - torch FD:", abs((f_autograd - f_fd_torch).item()))
        print("abs exported FD - torch FD:", abs((f_fd_exported - f_fd_torch).item()))
        print("abs torch FD - target:", abs((f_fd_torch - f_true).item()))
    finally:
        if not was_training:
            model.eval()


# -----------------------------------------------------------------------------
# Parity/evaluation
# -----------------------------------------------------------------------------


def exported_energy_parity_with_model(
    exported,
    model: BPNN,
    dataset,
    output_dir: Path,
    progress_interval: int,
) -> None:
    """Check exported-model energy parity with neighbor lists explicitly attached."""
    y_true = []
    y_pred = []
    samples = list(dataset)
    total = len(samples)
    start_time = time.time()
    print(f"energy parity: starting {total} systems", flush=True)

    for sample_i, sample in enumerate(samples, start=1):
        system = with_neighbor_lists(model, [copy.deepcopy(sample["system"])])[0]
        pred = exported([system], exported_energy_options(), check_consistency=True)
        y_true.append(true_energy(sample))
        y_pred.append(pred[ENERGY_TARGET].block(0).values.item())

        if progress_interval > 0 and (
            sample_i == 1 or sample_i == total or sample_i % progress_interval == 0
        ):
            print_progress("energy parity", sample_i, total, start_time)

    plot_parity(
        np.asarray(y_true),
        np.asarray(y_pred),
        xlabel="True energy / eV",
        ylabel="Predicted energy / eV",
        title="Energy parity, exported model",
        output_path=output_dir / "energy_parity_exported.png",
    )


def autograd_force_parity(
    model: BPNN,
    dataset,
    output_dir: Path,
    progress_interval: int,
) -> None:
    """Compute forces by differentiating the non-exported torch model."""
    f_true = []
    f_pred = []
    samples = list(dataset)
    total = len(samples)
    start_time = time.time()
    print(f"autograd force parity: starting {total} systems", flush=True)

    was_training = model.training
    model.train()
    try:
        for sample_i, sample in enumerate(samples):
            system = copy.deepcopy(sample["system"])
            system.positions.requires_grad_(True)
            system = with_neighbor_lists(model, [system])[0]

            model.zero_grad(set_to_none=True)
            if system.positions.grad is not None:
                system.positions.grad.zero_()

            energy = predict_energy_torch(model, system)
            forces_true = true_forces(sample).to(system.positions)

            if not energy.requires_grad:
                forces_pred = torch.zeros_like(forces_true)
            else:
                energy.backward()
                forces_pred = (
                    torch.zeros_like(forces_true)
                    if system.positions.grad is None
                    else -system.positions.grad.detach()
                )

            if sample_i < 5:
                print(
                    f"force autograd system {sample_i}: energy requires_grad={energy.requires_grad}",
                    flush=True,
                )

            f_true.append(forces_true.cpu().numpy().ravel())
            f_pred.append(forces_pred.cpu().numpy().ravel())

            done = sample_i + 1
            if progress_interval > 0 and (
                done == 1 or done == total or done % progress_interval == 0
            ):
                print_progress("autograd force parity", done, total, start_time)
    finally:
        if not was_training:
            model.eval()

    plot_parity(
        np.concatenate(f_true),
        np.concatenate(f_pred),
        xlabel="True force / eV Å$^{-1}$",
        ylabel="Autograd force / eV Å$^{-1}$",
        title="Force parity, non-exported autograd",
        output_path=output_dir / "force_parity_autograd.png",
    )


def finite_difference_force_component(
    exported, model: BPNN, system, atom_i: int, xyz: int, delta: float
) -> torch.Tensor:
    """Central finite-difference force component from exported energies."""
    plus = copy.deepcopy(system)
    minus = copy.deepcopy(system)
    plus.positions[atom_i, xyz] += delta
    minus.positions[atom_i, xyz] -= delta
    plus = with_neighbor_lists(model, [plus])[0]
    minus = with_neighbor_lists(model, [minus])[0]
    return -(
        predict_energy_exported(exported, plus).detach()
        - predict_energy_exported(exported, minus).detach()
    ) / (2.0 * delta)


def finite_difference_force_parity(
    exported,
    model: BPNN,
    dataset,
    output_dir: Path,
    delta: float,
    max_systems: int | None,
    progress_interval: int,
) -> None:
    """Compute finite-difference forces from exported model energies."""
    f_true = []
    f_pred = []
    samples = list(dataset)[:max_systems] if max_systems is not None else list(dataset)
    total = len(samples)
    start_time = time.time()
    print(f"finite-difference forces: starting {total} systems", flush=True)

    for sample_i, sample in enumerate(samples):
        system = copy.deepcopy(sample["system"])
        forces_true = true_forces(sample)
        forces_fd = torch.zeros_like(forces_true)

        for atom_i in range(system.positions.shape[0]):
            for xyz in range(3):
                forces_fd[atom_i, xyz] = finite_difference_force_component(
                    exported, model, system, atom_i, xyz, delta
                )

        f_true.append(forces_true.cpu().numpy().ravel())
        f_pred.append(forces_fd.cpu().numpy().ravel())

        done = sample_i + 1
        if progress_interval > 0 and (
            done == 1 or done == total or done % progress_interval == 0
        ):
            print_progress("finite-difference forces", done, total, start_time)

    plot_parity(
        np.concatenate(f_true),
        np.concatenate(f_pred),
        xlabel="True force / eV Å$^{-1}$",
        ylabel="Finite-difference force / eV Å$^{-1}$",
        title=f"Force parity, exported finite difference, delta={delta:g}",
        output_path=output_dir / "force_parity_finite_difference_exported.png",
    )


def finite_difference_torque_component(
    model: BPNN, system, atom_i: int, axis: int, delta: float
) -> torch.Tensor:
    """Central finite-difference torque component: tau = -dE/dtheta."""
    plus = copy.deepcopy(system)
    minus = copy.deepcopy(system)

    q0 = system.get_data(ANISOAP_QUATERNIONS).block(0).values
    q_plus = q0.clone()
    q_minus = q0.clone()

    dq_plus = small_rotation_quaternion(axis, +delta, dtype=q0.dtype, device=q0.device)
    dq_minus = small_rotation_quaternion(axis, -delta, dtype=q0.dtype, device=q0.device)

    # Left multiplication: lab/world-frame infinitesimal rotation.
    q_plus[atom_i] = quat_multiply(dq_plus, q_plus[atom_i])
    q_minus[atom_i] = quat_multiply(dq_minus, q_minus[atom_i])

    plus = replace_system_quaternions(plus, normalized_quaternions(q_plus))
    minus = replace_system_quaternions(minus, normalized_quaternions(q_minus))
    plus = with_neighbor_lists(model, [plus])[0]
    minus = with_neighbor_lists(model, [minus])[0]

    return -(
        predict_energy_torch(model, plus).detach()
        - predict_energy_torch(model, minus).detach()
    ) / (2.0 * delta)


def finite_difference_torque_parity(
    model: BPNN,
    dataset,
    output_dir: Path,
    delta: float,
    max_systems: int | None,
    progress_interval: int,
) -> None:
    """Evaluate model torques by finite-differencing energy w.r.t. rotations."""
    tau_true = []
    tau_pred = []
    samples = list(dataset)[:max_systems] if max_systems is not None else list(dataset)
    total = len(samples)
    start_time = time.time()
    print(f"finite-difference torques: starting {total} systems", flush=True)

    for sample_i, sample in enumerate(samples):
        system = copy.deepcopy(sample["system"])
        true = true_torques(sample).to(system.positions)
        pred = torch.zeros_like(true)

        for atom_i in range(system.positions.shape[0]):
            for axis in range(3):
                pred[atom_i, axis] = finite_difference_torque_component(
                    model, system, atom_i, axis, delta
                )

        tau_true.append(true.cpu().numpy().ravel())
        tau_pred.append(pred.cpu().numpy().ravel())

        done = sample_i + 1
        if progress_interval > 0 and (
            done == 1 or done == total or done % progress_interval == 0
        ):
            print_progress("finite-difference torques", done, total, start_time)

    plot_parity(
        np.concatenate(tau_true),
        np.concatenate(tau_pred),
        xlabel="True torque / eV",
        ylabel="Finite-difference torque / eV",
        title=f"Torque parity, finite difference, delta={delta:g}",
        output_path=output_dir / "torque_parity_finite_difference.png",
    )


# -----------------------------------------------------------------------------
# Training/splits/main
# -----------------------------------------------------------------------------


def split_dataset(
    dataset: Dataset, val_fraction: float, test_fraction: float, seed: int
):
    """Return deterministic train/val/test random splits."""
    n_total = len(dataset)
    n_test = max(1, int(round(test_fraction * n_total)))
    n_val = max(1, int(round(val_fraction * n_total)))
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"dataset too small for split: n_total={n_total}, "
            f"n_train={n_train}, n_val={n_val}, n_test={n_test}"
        )
    return random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )


def train_model(
    model: BPNN,
    train_dataset,
    val_dataset,
    output_dir: Path,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    force_weight: float,
) -> BPNN:
    """Train the model and return the model object that should be evaluated."""
    trainer = build_trainer(num_epochs, batch_size, learning_rate, force_weight)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(dtype=torch.float64)

    params_before = flattened_parameters(model)
    print("\n=== target sample labels debug ===")
    for i in range(min(8, len(train_dataset))):
        labels = train_dataset[i]["energy"].block().samples
        print(i, labels.names, labels.values.reshape(-1).tolist())
    print(
        f"trainer.train: starting epochs={num_epochs}, batch_size={batch_size}, "
        f"train={len(train_dataset)}, val={len(val_dataset)}",
        flush=True,
    )
    train_start = time.time()

    result = trainer.train(
        model=model,
        dtype=torch.float64,
        devices=[torch.device("cpu")],
        train_datasets=[train_dataset],
        val_datasets=[val_dataset],
        checkpoint_dir=str(output_dir),
    )

    print(f"trainer.train: finished in {time.time() - train_start:.1f}s", flush=True)

    if isinstance(result, BPNN):
        model = result
    elif isinstance(result, tuple):
        for item in result:
            if isinstance(item, BPNN):
                model = item
                break

    print_parameter_change(
        params_before, flattened_parameters(model), "after trainer.train"
    )
    return model


def evaluate_split(
    split_name: str,
    dataset,
    model: BPNN,
    exported,
    output_dir: Path,
    fd_delta: float,
    fd_max_systems: int | None,
    skip_energy: bool,
    skip_autograd: bool,
    skip_fd: bool,
    skip_torque: bool,
    progress_interval: int,
) -> None:
    """Run all requested parity checks for one split."""
    split_dir = output_dir / split_name
    print(f"\n=== evaluating split: {split_name} ({len(dataset)} systems) ===")

    if not skip_energy:
        exported_energy_parity_with_model(
            exported, model, dataset, split_dir, progress_interval
        )
    if not skip_autograd:
        autograd_force_parity(model, dataset, split_dir, progress_interval)
    if not skip_fd:
        finite_difference_force_parity(
            exported,
            model,
            dataset,
            split_dir,
            fd_delta,
            fd_max_systems,
            progress_interval,
        )
    if not skip_torque:
        finite_difference_torque_parity(
            model, dataset, split_dir, fd_delta, fd_max_systems, progress_interval
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AniSOAP-BPNN and run energy/force/torque checks."
    )
    parser.add_argument("--xyz", type=Path, default=Path("./all_frames.xyz"))
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("./model_outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--nmax", type=int, default=4)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--fd-delta", type=float, default=1e-4)
    parser.add_argument(
        "--fd-max-systems", type=int, default=2, help="Use -1 for all systems."
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-energy-parity", action="store_true")
    parser.add_argument("--skip-autograd", action="store_true")
    parser.add_argument("--skip-fd", action="store_true")
    parser.add_argument("--skip-torque", action="store_true")
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        choices=["train", "val", "test", "all"],
        default=["train", "test"],
        help="Dataset splits to evaluate.",
    )
    parser.add_argument(
        "--overfit-n",
        type=int,
        default=0,
        help="If >0, train and evaluate only the first N structures.",
    )
    parser.add_argument("--force-weight", type=float, default=10.0)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print progress every N systems in parity/evaluation loops. Use 0 to disable.",
    )
    parser.add_argument(
        "--quiet-trainer-logs",
        action="store_true",
        help="Disable INFO-level metatrain Trainer logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.quiet_trainer_logs:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
            force=True,
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_info = build_dataset_info()
    hypers = build_hypers(lmax=args.lmax, nmax=args.nmax, cutoff=args.cutoff)
    model = BPNN(hypers=hypers, dataset_info=dataset_info).to(dtype=torch.float64)

    print("target info:", dataset_info.targets[ENERGY_TARGET])
    print("reading:", args.xyz)
    print(
        "basis/cutoff:", {"lmax": args.lmax, "nmax": args.nmax, "cutoff": args.cutoff}
    )

    frames, systems = load_dataset(args.xyz, stride=args.stride)
    print_ase_frame_diagnostics(frames)

    dataset = build_dataset(frames, systems)

    if args.overfit_n > 0:
        dataset = torch.utils.data.Subset(dataset, list(range(args.overfit_n)))
        train_dataset = dataset
        val_dataset = dataset
        test_dataset = dataset
    else:
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )

    print("loaded systems:", len(dataset))
    print(
        "split sizes:",
        {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
    )

    sample = dataset[0]
    print("target gradients:", sample[ENERGY_TARGET].block(0).gradients_list())
    print("force target norm:", true_forces(sample).norm().item())
    print("torque target norm:", true_torques(sample).norm().item())

    print_model_diagnostics(model)
    check_single_displacement_sensitivity(model, sample, delta=1e-3)
    check_single_rotation_sensitivity(model, sample, delta=1e-3)
    check_single_component_force_consistency(model, sample, delta=args.fd_delta)

    if args.diagnostics_only:
        print("diagnostics-only requested; stopping before training")
        return

    if not args.skip_train:
        model = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            force_weight=args.force_weight,
        )
        print("\n=== post-training model diagnostics ===")
        print_model_diagnostics(model)
        check_single_component_force_consistency(model, sample, delta=args.fd_delta)

    exported = copy.deepcopy(model).export()
    max_systems = None if args.fd_max_systems < 0 else args.fd_max_systems

    split_lookup = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "all": dataset,
    }
    requested_splits = (
        ["train", "val", "test"] if "all" in args.eval_splits else args.eval_splits
    )

    for split_name in requested_splits:
        evaluate_split(
            split_name=split_name,
            dataset=split_lookup[split_name],
            model=model,
            exported=exported,
            output_dir=args.output_dir,
            fd_delta=args.fd_delta,
            fd_max_systems=max_systems,
            skip_energy=args.skip_energy_parity,
            skip_autograd=args.skip_autograd,
            skip_fd=args.skip_fd,
            skip_torque=args.skip_torque,
            progress_interval=args.progress_interval,
        )

    print("done")
    print(f"plots written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
