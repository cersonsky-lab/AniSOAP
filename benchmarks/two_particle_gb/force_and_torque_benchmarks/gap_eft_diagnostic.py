from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatrain.anisoap_gap import AnisoapGAP as GAP
from metatrain.anisoap_gap import Trainer
from metatrain.anisoap_gap.trainer import (
    _apply_rotation_vector_parameters,
    _get_quaternion_values,
    clone_system_with_position_leaf,
)
from metatrain.utils.data import Dataset
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from anisoap_checks import (
    ENERGY_TARGET,
    build_dataset,
    build_dataset_info,
    load_dataset,
    plot_parity,
    true_energy,
    true_forces,
    true_torques,
    with_neighbor_lists,
)
from orientation_sweep_r2p4_check import gap_training_hypers, hypers as gap_hypers


torch.set_default_dtype(torch.float64)


def vector_target(
    values: torch.Tensor,
    *,
    property_name: str,
    sample_i: int | None = None,
) -> TensorMap:
    """Build an explicit per-atom vector target with values shaped [n_atoms, 3, 1].

    If sample_i is provided, sample labels are globally unique across frames, which
    is required by mts.join(..., axis="samples") inside the GAP trainer.
    """
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.ndim == 2:
        values = values.reshape(values.shape[0], 3, 1)

    if values.ndim != 3 or values.shape[1:] != (3, 1):
        raise ValueError(f"expected vector target [n_atoms, 3, 1], got {tuple(values.shape)}")

    n_atoms = values.shape[0]

    if sample_i is None:
        sample_names = ["atom"]
        sample_values = torch.arange(n_atoms, dtype=torch.int32).reshape(-1, 1)
    else:
        sample_names = ["sample", "atom"]
        sample_values = torch.stack(
            [
                torch.full((n_atoms,), int(sample_i), dtype=torch.int32),
                torch.arange(n_atoms, dtype=torch.int32),
            ],
            dim=1,
        )

    block = TensorBlock(
        values=values,
        samples=Labels(sample_names, sample_values),
        components=[Labels(["xyz"], torch.tensor([[0], [1], [2]], dtype=torch.int32))],
        properties=Labels([property_name], torch.tensor([[0]], dtype=torch.int32)),
    )
    return TensorMap(keys=Labels.single(), blocks=[block])



def scalar_energy_target(value: float, *, sample_i: int | None = None) -> TensorMap:
    """Build scalar energy target without gradients.

    If sample_i is provided, sample labels are globally unique across frames.
    """
    if sample_i is None:
        samples = Labels(["system"], torch.tensor([[0]], dtype=torch.int32))
    else:
        samples = Labels(["sample"], torch.tensor([[int(sample_i)]], dtype=torch.int32))

    block = TensorBlock(
        values=torch.tensor([[float(value)]], dtype=torch.float64),
        samples=samples,
        components=[],
        properties=Labels(["energy"], torch.tensor([[0]], dtype=torch.int32)),
    )
    return TensorMap(keys=Labels.single(), blocks=[block])


def gap_dataset_with_explicit_eft_targets(dataset) -> Dataset:
    """
    Convert the existing AniSOAP-BPNN diagnostic dataset to the explicit target
    layout used by the degree-one GAP E/F/T linear-system trainer.

    Existing helpers already build:
      - energy target
      - position gradients on energy target, from which true_forces(...) reads forces
      - explicit torques target

    GAP E/F/T training currently expects:
      - energy
      - forces
      - torques
    """
    systems = []
    energies = []
    forces = []
    torques = []

    for sample_i, sample in enumerate(dataset):
        systems.append(sample["system"])
        energies.append(scalar_energy_target(true_energy(sample), sample_i=sample_i))
        forces.append(
            vector_target(
                true_forces(sample),
                property_name="force",
                sample_i=sample_i,
            )
        )
        torques.append(
            vector_target(
                true_torques(sample),
                property_name="torque",
                sample_i=sample_i,
            )
        )

    return Dataset.from_dict(
        {
            "system": systems,
            ENERGY_TARGET: energies,
            "forces": forces,
            "torques": torques,
        }
    )


def check_gap_dataset(dataset) -> None:
    print("GAP E/F/T dataset readiness")
    print("---------------------------")
    print("n_samples:", len(dataset))

    n_atoms_total = 0
    for sample_i, sample in enumerate(dataset):
        system = sample["system"]
        n_atoms = len(system.types)
        n_atoms_total += n_atoms

        q = _get_quaternion_values(system)
        if tuple(q.shape) != (n_atoms, 4):
            raise ValueError(
                f"sample {sample_i}: quaternion shape {tuple(q.shape)} != {(n_atoms, 4)}"
            )
        if not torch.isfinite(q).all():
            raise ValueError(f"sample {sample_i}: quaternions contain non-finite values")

        for target_name in ["forces", "torques"]:
            values = sample[target_name].block(0).values
            expected = (n_atoms, 3, 1)
            if tuple(values.shape) != expected:
                raise ValueError(
                    f"sample {sample_i}: {target_name} shape {tuple(values.shape)} != {expected}"
                )
            if not torch.isfinite(values).all():
                raise ValueError(f"sample {sample_i}: {target_name} contains non-finite values")

    print("n_atoms_total:", n_atoms_total)
    print("readiness: OK")
    print()


def predict_gap_energy_force_torque(model: GAP, system):
    """
    Conservative GAP prediction:
      energy = model(system)
      forces = -dE/dR
      torques = -dE/d(delta_rotation)
    """
    system_i, _positions_leaf = clone_system_with_position_leaf(copy.deepcopy(system))

    q = _get_quaternion_values(system_i)
    delta_theta = torch.zeros(
        (q.shape[0], 3),
        dtype=q.dtype,
        device=q.device,
        requires_grad=True,
    )

    system_i = _apply_rotation_vector_parameters(system_i, delta_theta)
    system_i = get_system_with_neighbor_lists(
        system_i,
        get_requested_neighbor_lists(model),
    )

    energy = model([system_i], model.outputs)[ENERGY_TARGET].block(0).values.reshape(())

    grad_pos, grad_theta = torch.autograd.grad(
        energy,
        [system_i.positions, delta_theta],
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )

    if grad_pos is None:
        grad_pos = torch.zeros_like(system_i.positions)
    if grad_theta is None:
        grad_theta = torch.zeros_like(delta_theta)

    return (
        energy.reshape(1, 1),
        (-grad_pos).reshape(-1, 3, 1),
        (-grad_theta).reshape(-1, 3, 1),
    )


def joined_values(dataset, name: str):
    """Concatenate target values without requiring globally unique sample labels."""
    return torch.cat(
        [sample[name].block(0).values for sample in dataset],
        dim=0,
    )


def rmse(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - ref) ** 2)).detach())


def rel_rmse(pred: torch.Tensor, ref: torch.Tensor) -> float:
    denom = torch.sqrt(torch.mean((ref - ref.mean()) ** 2)).detach().clamp_min(1e-12)
    return float((torch.sqrt(torch.mean((pred - ref) ** 2)) / denom).detach())



def collect_gap_eft_predictions(model: GAP, dataset):
    e_pred = []
    f_pred = []
    t_pred = []

    for sample in dataset:
        e_i, f_i, t_i = predict_gap_energy_force_torque(model, sample["system"])
        e_pred.append(e_i)
        f_pred.append(f_i)
        t_pred.append(t_i)

    return (
        torch.cat(e_pred, dim=0),
        torch.cat(f_pred, dim=0),
        torch.cat(t_pred, dim=0),
    )


def gap_eft_metrics(model: GAP, dataset):
    e_pred, f_pred, t_pred = collect_gap_eft_predictions(model, dataset)

    e_ref = joined_values(dataset, ENERGY_TARGET).reshape(-1, 1)
    f_ref = joined_values(dataset, "forces")
    t_ref = joined_values(dataset, "torques")

    return {
        "energy_rmse": rmse(e_pred, e_ref),
        "energy_rel_rmse": rel_rmse(e_pred, e_ref),
        "forces_rmse": rmse(f_pred, f_ref),
        "forces_rel_rmse": rel_rmse(f_pred, f_ref),
        "torques_rmse": rmse(t_pred, t_ref),
        "torques_rel_rmse": rel_rmse(t_pred, t_ref),
    }, (e_ref, f_ref, t_ref), (e_pred, f_pred, t_pred)


def evaluate_gap_eft(
    model: GAP,
    dataset,
    output_dir: Path,
    *,
    metadata: dict | None = None,
) -> None:
    metrics, refs, preds = gap_eft_metrics(model, dataset)
    e_ref, f_ref, t_ref = refs
    e_pred, f_pred, t_pred = preds

    print("GAP E/F/T metrics")
    print("-----------------")
    print(f"energy RMSE:      {metrics['energy_rmse']:.8e}")
    print(f"energy rel RMSE:  {metrics['energy_rel_rmse']:.8e}")
    print(f"forces RMSE:      {metrics['forces_rmse']:.8e}")
    print(f"forces rel RMSE:  {metrics['forces_rel_rmse']:.8e}")
    print(f"torques RMSE:     {metrics['torques_rmse']:.8e}")
    print(f"torques rel RMSE: {metrics['torques_rel_rmse']:.8e}")

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_parity(
        e_ref.detach().cpu().numpy().reshape(-1),
        e_pred.detach().cpu().numpy().reshape(-1),
        xlabel="True energy / eV",
        ylabel="GAP energy / eV",
        title="GAP degree=1 linear-system E/F/T: energy",
        output_path=output_dir / "gap_energy_parity.png",
    )
    plot_parity(
        f_ref.detach().cpu().numpy().reshape(-1),
        f_pred.detach().cpu().numpy().reshape(-1),
        xlabel="True force",
        ylabel="GAP force",
        title="GAP degree=1 linear-system E/F/T: forces",
        output_path=output_dir / "gap_forces_parity.png",
    )
    plot_parity(
        t_ref.detach().cpu().numpy().reshape(-1),
        t_pred.detach().cpu().numpy().reshape(-1),
        xlabel="True torque",
        ylabel="GAP torque",
        title="GAP degree=1 linear-system E/F/T: torques",
        output_path=output_dir / "gap_torques_parity.png",
    )

    payload = {
        "metrics": metrics,
        "metadata": metadata or {},
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print()
    print("wrote plots and metrics to", output_dir)


def print_reference_ranges(dataset) -> None:
    energies = joined_values(dataset, ENERGY_TARGET).reshape(-1)
    forces = joined_values(dataset, "forces").reshape(-1)
    torques = joined_values(dataset, "torques").reshape(-1)

    print("Reference target ranges")
    print("-----------------------")
    print(
        "energy min/max/spread:",
        float(energies.min()),
        float(energies.max()),
        float(energies.max() - energies.min()),
    )
    print(
        "force min/max/spread:",
        float(forces.min()),
        float(forces.max()),
        float(forces.max() - forces.min()),
    )
    print(
        "torque min/max/spread:",
        float(torques.min()),
        float(torques.max()),
        float(torques.max() - torques.min()),
    )
    print()


def subset_dataset(dataset, indices):
    return Dataset.from_dict(
        {
            "system": [dataset[i]["system"] for i in indices],
            ENERGY_TARGET: [dataset[i][ENERGY_TARGET] for i in indices],
            "forces": [dataset[i]["forces"] for i in indices],
            "torques": [dataset[i]["torques"] for i in indices],
        }
    )


def train_test_indices(n_samples: int, test_every: int | None):
    all_indices = list(range(n_samples))

    if test_every is None or test_every <= 0:
        return all_indices, []

    test_indices = [i for i in all_indices if i % test_every == 0]
    train_indices = [i for i in all_indices if i % test_every != 0]

    if not train_indices:
        raise ValueError("--test-every selected all frames for test; no training frames remain.")

    if not test_indices:
        raise ValueError("--test-every produced no test frames.")

    return train_indices, test_indices


def train_gap_eft_model(
    gap_dataset,
    *,
    lmax: int,
    nmax: int,
    num_sparse_points: int,
    regularizer: float,
    energy_weight: float,
    force_weight: float,
    torque_weight: float,
    normalize_targets: bool,
):
    model = GAP(
        gap_hypers(
            num_sparse_points=min(num_sparse_points, len(gap_dataset)),
            degree=1,
            lmax=lmax,
            nmax=nmax,
            regularizer=regularizer,
        ),
        build_dataset_info(add_forces=False),
    )

    trainer = Trainer(
        gap_training_hypers(
            regularizer,
            energy_weight=energy_weight,
            force_weight=force_weight,
            torque_weight=torque_weight,
            normalize_targets=normalize_targets,
            enable_force_torque_training=True,
        )
    )

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        trainer.train(
            model=model,
            dtype=torch.float64,
            devices=[torch.device("cpu")],
            train_datasets=[gap_dataset],
            val_datasets=[gap_dataset],
            checkpoint_dir=checkpoint_dir,
        )

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", default="orientation_sweep_r2p4.xyz")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", default="gap_eft_diagnostic_out")
    parser.add_argument("--lmax", type=int, default=5)
    parser.add_argument("--nmax", type=int, default=5)
    parser.add_argument("--num-sparse-points", type=int, default=8)
    parser.add_argument("--regularizer", type=float, default=1e-8)
    parser.add_argument("--energy-weight", type=float, default=10.0)
    parser.add_argument("--force-weight", type=float, default=1.0)
    parser.add_argument("--torque-weight", type=float, default=1.0)
    parser.add_argument("--no-normalize-targets", action="store_true")
    parser.add_argument(
        "--test-every",
        type=int,
        default=None,
        help="Hold out every Nth frame for test evaluation. Example: --test-every 4.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a small E/F/T weight sweep instead of a single diagnostic.",
    )
    args = parser.parse_args()

    xyz = Path(args.xyz).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    frames, systems = load_dataset(
        xyz,
        stride=args.stride,
    )
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
        systems = systems[: args.max_frames]

    bpnn_style_dataset = build_dataset(frames, systems, add_forces=True)
    gap_dataset = gap_dataset_with_explicit_eft_targets(bpnn_style_dataset)

    check_gap_dataset(gap_dataset)
    print_reference_ranges(gap_dataset)

    train_indices, test_indices = train_test_indices(len(gap_dataset), args.test_every)
    train_dataset = subset_dataset(gap_dataset, train_indices)
    test_dataset = subset_dataset(gap_dataset, test_indices) if test_indices else None

    if test_indices:
        print("Train/test split")
        print("----------------")
        print("train frames:", train_indices)
        print("test frames:", test_indices)
        print()

    if args.sweep:
        sweep = [
            (10.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 3.0),
            (1.0, 1.0, 10.0),
            (1.0, 0.3, 10.0),
            (0.3, 0.3, 10.0),
            (0.1, 0.3, 10.0),
            (0.1, 0.1, 30.0),
        ]

        print("Weight sweep")
        print("------------")
        print("E_w F_w T_w | E_RMSE F_RMSE T_RMSE | E_rel F_rel T_rel")

        best = None
        for e_w, f_w, t_w in sweep:
            model = train_gap_eft_model(
                train_dataset,
                lmax=args.lmax,
                nmax=args.nmax,
                num_sparse_points=args.num_sparse_points,
                regularizer=args.regularizer,
                energy_weight=e_w,
                force_weight=f_w,
                torque_weight=t_w,
                normalize_targets=not args.no_normalize_targets,
            )
            metrics, _, _ = gap_eft_metrics(model, train_dataset)
            score = (
                metrics["energy_rel_rmse"]
                + metrics["forces_rel_rmse"]
                + metrics["torques_rel_rmse"]
            )

            print(
                f"{e_w:4.1f} {f_w:4.1f} {t_w:4.1f} | "
                f"{metrics['energy_rmse']:.3e} {metrics['forces_rmse']:.3e} {metrics['torques_rmse']:.3e} | "
                f"{metrics['energy_rel_rmse']:.3e} {metrics['forces_rel_rmse']:.3e} {metrics['torques_rel_rmse']:.3e}"
            )

            if best is None or score < best[0]:
                best = (score, e_w, f_w, t_w, model, metrics)

        assert best is not None
        score, e_w, f_w, t_w, model, metrics = best
        print()
        print(
            f"best weights: energy={e_w}, force={f_w}, torque={t_w}, "
            f"score={score:.6e}"
        )
        evaluate_gap_eft(
            model,
            train_dataset,
            output_dir / f"best_E{e_w:g}_F{f_w:g}_T{t_w:g}_train",
            metadata={
                "mode": "sweep_best_train",
                "energy_weight": e_w,
                "force_weight": f_w,
                "torque_weight": t_w,
                "score": score,
                "lmax": args.lmax,
                "nmax": args.nmax,
                "num_sparse_points": args.num_sparse_points,
                "regularizer": args.regularizer,
                "normalize_targets": not args.no_normalize_targets,
                "xyz": str(xyz),
                "stride": args.stride,
                "max_frames": args.max_frames,
            },
        )
        if test_dataset is not None:
            evaluate_gap_eft(
                model,
                test_dataset,
                output_dir / f"best_E{e_w:g}_F{f_w:g}_T{t_w:g}_test",
                metadata={
                    "mode": "sweep_best_test",
                    "energy_weight": e_w,
                    "force_weight": f_w,
                    "torque_weight": t_w,
                    "score": score,
                    "lmax": args.lmax,
                    "nmax": args.nmax,
                    "num_sparse_points": args.num_sparse_points,
                    "regularizer": args.regularizer,
                    "normalize_targets": not args.no_normalize_targets,
                    "xyz": str(xyz),
                    "stride": args.stride,
                    "max_frames": args.max_frames,
                    "test_every": args.test_every,
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                },
            )
        return

    model = train_gap_eft_model(
        gap_dataset,
        lmax=args.lmax,
        nmax=args.nmax,
        num_sparse_points=args.num_sparse_points,
        regularizer=args.regularizer,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        torque_weight=args.torque_weight,
        normalize_targets=not args.no_normalize_targets,
    )

    evaluate_gap_eft(
        model,
        train_dataset,
        output_dir / ("train" if test_dataset is not None else "."),
        metadata={
            "mode": "single_train" if test_dataset is not None else "single",
            "energy_weight": args.energy_weight,
            "force_weight": args.force_weight,
            "torque_weight": args.torque_weight,
            "lmax": args.lmax,
            "nmax": args.nmax,
            "num_sparse_points": args.num_sparse_points,
            "regularizer": args.regularizer,
            "normalize_targets": not args.no_normalize_targets,
            "xyz": str(xyz),
            "stride": args.stride,
            "max_frames": args.max_frames,
            "test_every": args.test_every,
            "train_indices": train_indices,
            "test_indices": test_indices,
        },
    )

    if test_dataset is not None:
        evaluate_gap_eft(
            model,
            test_dataset,
            output_dir / "test",
            metadata={
                "mode": "single_test",
                "energy_weight": args.energy_weight,
                "force_weight": args.force_weight,
                "torque_weight": args.torque_weight,
                "lmax": args.lmax,
                "nmax": args.nmax,
                "num_sparse_points": args.num_sparse_points,
                "regularizer": args.regularizer,
                "normalize_targets": not args.no_normalize_targets,
                "xyz": str(xyz),
                "stride": args.stride,
                "max_frames": args.max_frames,
                "test_every": args.test_every,
                "train_indices": train_indices,
                "test_indices": test_indices,
            },
        )


if __name__ == "__main__":
    main()
