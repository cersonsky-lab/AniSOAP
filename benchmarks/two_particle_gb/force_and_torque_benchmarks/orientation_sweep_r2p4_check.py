import argparse
import copy
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read
from metatomic.torch import systems_to_torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatrain.anisoap_gap import AnisoapGAP as GAP
from metatrain.anisoap_gap import Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

torch.set_default_dtype(torch.float64)


def hypers(num_sparse_points, degree, lmax, nmax, regularizer):
    return {
        "soap": {
            "cutoff": {
                "radius": 5.0,
                "smoothing": {"type": "ShiftedCosine", "width": 0.5},
                "width": 0.5,
            },
            "density": {"type": "Gaussian", "width": 1.0},
            "basis": {
                "type": "TensorProduct",
                "max_angular": lmax,
                "radial": {"type": "Gto", "max_radial": nmax},
            },
            "max_angular": lmax,
            "max_radial": nmax,
        },
        "krr": {
            "degree": degree,
            "num_sparse_points": num_sparse_points,
            "full_sparse_points": True,
        },
        "zbl": False,
        "aggregate_names": ["atom", "center_type"],
    }


def training_hypers(regularizer):
    return {
        "regularizer": float(regularizer),
        "regularizer_forces": float(regularizer),
        "atomic_baseline": {},
    }


def dataset_info(atomic_types):
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=atomic_types,
        targets={
            "energy": get_energy_target_info(
                "energy",
                {"quantity": "energy", "unit": "eV"},
                add_position_gradients=False,
            )
        },
    )


def per_atom_tensormap(array, property_name):
    values = torch.as_tensor(array, dtype=torch.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    n_atoms, n_properties = values.shape
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=values,
                samples=Labels(
                    ["atom"],
                    torch.arange(n_atoms, dtype=torch.int32).reshape(-1, 1),
                ),
                components=[],
                properties=Labels(
                    [property_name],
                    torch.arange(n_properties, dtype=torch.int32).reshape(-1, 1),
                ),
            )
        ],
    )


def get_energy(atoms):
    # Try common ASE/extended-XYZ locations.
    for key in ["energy", "Energy", "E", "free_energy"]:
        if key in atoms.info:
            return float(atoms.info[key])

    try:
        return float(atoms.get_potential_energy())
    except Exception as exc:
        raise RuntimeError(
            f"Could not find energy in atoms.info keys={list(atoms.info.keys())}"
        ) from exc


def get_quaternions(atoms):
    # Common shapes:
    # atoms.arrays["quaternions"] -> (n_atoms, 4)
    # atoms.arrays["anisoap::quaternions"] may be unavailable in ASE because ':' is awkward.
    for key in [
        "quaternions",
        "quaternion",
        "anisoap_quaternions",
        "anisoap__quaternions",
    ]:
        if key in atoms.arrays:
            q = np.asarray(atoms.arrays[key], dtype=np.float64)
            if q.shape[1] == 4:
                return q

    # Split columns q0/q1/q2/q3 or quat0..quat3.
    for prefix in ["q", "quat", "quaternion"]:
        keys = [f"{prefix}{i}" for i in range(4)]
        if all(k in atoms.arrays for k in keys):
            return np.stack(
                [np.asarray(atoms.arrays[k], dtype=np.float64) for k in keys], axis=1
            )

    raise RuntimeError(
        f"Could not find quaternion arrays. arrays={list(atoms.arrays.keys())}"
    )


def get_lengths(atoms):
    # Prefer semiaxes if present.
    for key in [
        "ellipsoid_lengths",
        "anisoap_ellipsoid_lengths",
        "anisoap__ellipsoid_lengths",
    ]:
        if key in atoms.arrays:
            x = np.asarray(atoms.arrays[key], dtype=np.float64)
            if x.shape[1] == 3:
                return x

    # Common diameter columns; convert diameters -> semiaxis lengths.
    diameter_names = [
        ("c_diameter[1]", "c_diameter[2]", "c_diameter[3]"),
        ("c_diameter1", "c_diameter2", "c_diameter3"),
        ("anisoap_c_diameter_1", "anisoap_c_diameter_2", "anisoap_c_diameter_3"),
    ]
    for names in diameter_names:
        if all(name in atoms.arrays for name in names):
            diam = np.stack(
                [np.asarray(atoms.arrays[name], dtype=np.float64) for name in names],
                axis=1,
            )
            return 0.5 * diam

    # If every frame has same ellipsoid lengths but no arrays, use the toy default.
    print("WARNING: no lengths found; using [0.5, 0.75, 1.0] for all atoms")
    return np.tile(np.array([[0.5, 0.75, 1.0]], dtype=np.float64), (len(atoms), 1))


def atoms_to_system(atoms):
    atoms = atoms.copy()
    atoms.cell = [0.0, 0.0, 0.0]
    atoms.pbc = False
    system = systems_to_torch([atoms], dtype=torch.float64)[0]

    q = get_quaternions(atoms)
    lengths = get_lengths(atoms)

    system.add_data("anisoap::quaternions", per_atom_tensormap(q, "q"))
    system.add_data("anisoap::ellipsoid_lengths", per_atom_tensormap(lengths, "axis"))

    diam = 2.0 * lengths
    for axis in range(3):
        system.add_data(
            f"anisoap::c_diameter_{axis + 1}",
            per_atom_tensormap(diam[:, axis], "c"),
        )

    return system


def energy_target(value, system_i):
    block = TensorBlock(
        values=torch.tensor([[value]], dtype=torch.float64),
        samples=Labels(["system"], torch.tensor([[system_i]], dtype=torch.int32)),
        components=[],
        properties=Labels(["energy"], torch.tensor([[0]], dtype=torch.int32)),
    )
    return TensorMap(keys=Labels.single(), blocks=[block])


def load_dataset(xyz_path, stride=1, max_frames=None):
    atoms_list = read(xyz_path, index=":")
    atoms_list = atoms_list[::stride]
    if max_frames is not None:
        atoms_list = atoms_list[:max_frames]

    systems = []
    targets = []
    energies = []
    for i, atoms in enumerate(atoms_list):
        e = get_energy(atoms)
        systems.append(atoms_to_system(atoms))
        targets.append(energy_target(e, i))
        energies.append(e)

    return Dataset.from_dict({"system": systems, "energy": targets}), np.array(energies)


def with_neighbor_lists(model, systems):
    requested = get_requested_neighbor_lists(model)
    return [get_system_with_neighbor_lists(system, requested) for system in systems]


def predict_gap(model, dataset):
    systems = with_neighbor_lists(
        model,
        [copy.deepcopy(dataset[i]["system"]) for i in range(len(dataset))],
    )
    model._subset_of_regressors_torch = (
        model._subset_of_regressors.export_torch_script_model()
    )
    model.eval()
    pred = model(systems, {"energy": model.outputs["energy"]})["energy"]
    return pred.block().values.detach().cpu().numpy().reshape(-1)


def system_feature_matrix(model, dataset):
    systems = with_neighbor_lists(
        model,
        [copy.deepcopy(dataset[i]["system"]) for i in range(len(dataset))],
    )
    features = model._soap_torch_calculator.compute(systems)

    block = features.block(0)
    values = block.values
    sample_systems = block.samples.values[:, 0]

    rows = []
    for system_i in range(len(dataset)):
        rows.append(
            values[sample_systems == system_i]
            .sum(dim=0)
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
        )

    return np.vstack(rows)


def ridge_fit_predict(x_train, y_train, x_eval, alpha):
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_scale = x_train.std(axis=0, keepdims=True)
    x_scale[x_scale < 1e-12] = 1.0

    xt = (x_train - x_mean) / x_scale
    xe = (x_eval - x_mean) / x_scale

    at = np.hstack([np.ones((xt.shape[0], 1)), xt])
    ae = np.hstack([np.ones((xe.shape[0], 1)), xe])

    reg = alpha * np.eye(at.shape[1])
    reg[0, 0] = 0.0
    w = np.linalg.solve(at.T @ at + reg, at.T @ y_train)
    return ae @ w


def rmse(y, yhat):
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def subset_dataset(dataset, indices):
    return Dataset.from_dict(
        {
            "system": [dataset[int(i)]["system"] for i in indices],
            "energy": [dataset[int(i)]["energy"] for i in indices],
        }
    )


def gap_style_system_feature_matrix(model, dataset):
    """System feature matrix using the same feature preprocessing as AniSOAP-GAP."""
    systems = with_neighbor_lists(
        model,
        [copy.deepcopy(dataset[i]["system"]) for i in range(len(dataset))],
    )

    features = model._soap_torch_calculator.compute(systems)
    features = features.keys_to_samples("center_type")
    if (
        "neighbor_1_type" in features.keys.names
        and "neighbor_2_type" in features.keys.names
    ):
        features = features.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])

    # Match GAP normalization.
    rows = []
    block = features.block()
    values = block.values
    norms = np.linalg.norm(values.detach().cpu().numpy(), axis=1, keepdims=True)
    norms[norms < 1e-30] = 1.0
    values_np = values.detach().cpu().numpy() / norms

    sample_names = list(block.samples.names)
    system_col = sample_names.index("system")
    sample_systems = block.samples.values[:, system_col].detach().cpu().numpy()

    for system_i in range(len(dataset)):
        rows.append(values_np[sample_systems == system_i].sum(axis=0).reshape(-1))

    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", default="orientation_sweep_r2p4.xyz")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--train-n", type=int, default=64)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--sparse", type=int, default=32)
    parser.add_argument("--regularizer", type=float, default=1e-10)
    parser.add_argument("--ridge-alpha", type=float, default=1e-8)
    parser.add_argument("--lmax", type=int, default=8)
    parser.add_argument("--nmax", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", default="orientation_sweep_r2p4_gap_check.png")
    args = parser.parse_args()

    data, y = load_dataset(args.xyz, stride=args.stride, max_frames=args.max_frames)

    atoms0 = read(args.xyz, index=0)
    atomic_types = sorted(set(int(z) for z in atoms0.numbers))
    print("atomic_types:", atomic_types)

    rng = np.random.default_rng(args.seed)
    n_total = len(data)
    train_n = min(args.train_n, n_total)

    if train_n == n_total:
        train_idx = np.arange(n_total)
    else:
        train_idx = np.sort(rng.choice(n_total, size=train_n, replace=False))

    eval_idx = np.arange(n_total)

    train_data = subset_dataset(data, train_idx)
    eval_data = data

    y_train = y[train_idx]
    y_eval = y[eval_idx]

    model = GAP(
        hypers(
            num_sparse_points=min(args.sparse, train_n),
            degree=args.degree,
            lmax=args.lmax,
            nmax=args.nmax,
            regularizer=args.regularizer,
        ),
        dataset_info(atomic_types),
    )

    trainer = Trainer(training_hypers(args.regularizer))
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        trainer.train(
            model=model,
            dtype=torch.float64,
            devices=[torch.device("cpu")],
            train_datasets=[train_data],
            val_datasets=[train_data],
            checkpoint_dir=checkpoint_dir,
        )

    pred_train = predict_gap(model, train_data)
    pred_eval = predict_gap(model, eval_data)

    # Bare linear ridge on same AniSOAP features.
    x_train = system_feature_matrix(model, train_data)
    x_eval = system_feature_matrix(model, eval_data)
    pred_ridge_eval = ridge_fit_predict(
        x_train, y_train, x_eval, alpha=args.ridge_alpha
    )

    # GAP-style linear ridge on the same preprocessed features GAP uses.
    x_gapstyle_train = gap_style_system_feature_matrix(model, train_data)
    x_gapstyle_eval = gap_style_system_feature_matrix(model, eval_data)
    pred_gapstyle_ridge_eval = ridge_fit_predict(
        x_gapstyle_train,
        y_train,
        x_gapstyle_eval,
        alpha=args.ridge_alpha,
    )
    print("GAP-style linear ridge eval RMSE:", rmse(y_eval, pred_gapstyle_ridge_eval))
    print(
        "GAP-style linear ridge eval range:",
        float(pred_gapstyle_ridge_eval.max() - pred_gapstyle_ridge_eval.min()),
    )

    baseline_train = np.full_like(y_train, y_train.mean())
    baseline_eval = np.full_like(y_eval, y_train.mean())

    print("frames:", n_total)
    print("train_n:", train_n)
    print(
        "energy range:",
        float(y_eval.min()),
        float(y_eval.max()),
        "spread",
        float(y_eval.max() - y_eval.min()),
    )
    print("GAP train RMSE:", rmse(y_train, pred_train))
    print("constant train RMSE:", rmse(y_train, baseline_train))
    print("GAP eval RMSE:", rmse(y_eval, pred_eval))
    print("linear ridge eval RMSE:", rmse(y_eval, pred_ridge_eval))
    print("constant eval RMSE:", rmse(y_eval, baseline_eval))
    print("GAP eval range:", float(pred_eval.max() - pred_eval.min()))
    print(
        "linear ridge eval range:", float(pred_ridge_eval.max() - pred_ridge_eval.min())
    )
    print("target eval range:", float(y_eval.max() - y_eval.min()))

    order = np.argsort(y_eval)
    x = np.arange(n_total)

    plt.figure(figsize=(9, 5))
    plt.scatter(y_eval, pred_ridge_eval, label="bare linear AniSOAP ridge")
    plt.scatter(y_eval, pred_gapstyle_ridge_eval, label="GAP-style ridge")
    plt.scatter(y_eval, pred_eval, label=f"AniSOAP-GAP degree={args.degree}")

    plt.title("orientation_sweep_r2p4 energy-only diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot, dpi=200)
    print("wrote", args.plot)


if __name__ == "__main__":
    main()
