from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read
from anisoap.representations import EllipsoidalDensityProjection
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


torch.set_default_dtype(torch.float64)


def load_frames_and_energies(
    path: Path,
    stride: int,
    max_frames: int | None,
):
    """
    Load ASE frames for direct AniSOAP calculation and obtain analytical
    benchmark energies from the corresponding benchmark dataset samples.

    This uses no GAP model and no metatrain feature machinery.
    """
    from anisoap_checks import (
        build_dataset,
        load_dataset,
        true_energy,
    )

    frames, systems = load_dataset(path, stride=stride)

    if max_frames is not None:
        frames = frames[:max_frames]
        systems = systems[:max_frames]

    if not frames:
        raise ValueError("No frames were loaded.")

    for i, atoms in enumerate(frames):
        if "quaternions" not in atoms.arrays:
            raise KeyError(f"frame {i} is missing arrays['quaternions']")

        for key in ("c_diameter[1]", "c_diameter[2]", "c_diameter[3]"):
            if key not in atoms.arrays:
                raise KeyError(f"frame {i} is missing arrays[{key!r}]")

    benchmark_dataset = build_dataset(
        frames,
        systems,
        add_forces=False,
        add_torques=False,
    )

    energies = np.asarray(
        [
            float(true_energy(benchmark_dataset[i]))
            for i in range(len(benchmark_dataset))
        ],
        dtype=np.float64,
    )

    if energies.shape != (len(frames),):
        raise ValueError(
            f"Expected one energy per frame; got shape {energies.shape} "
            f"for {len(frames)} frames."
        )

    return frames, energies





def split_indices(n_frames: int, test_every: int | None):
    if test_every is None or test_every <= 0:
        return np.arange(n_frames), np.array([], dtype=int)

    test_idx = np.arange(0, n_frames, test_every)
    mask = np.ones(n_frames, dtype=bool)
    mask[test_idx] = False
    train_idx = np.arange(n_frames)[mask]
    return train_idx, test_idx


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rel_rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    denom = np.sqrt(np.mean((y_true - y_true.mean()) ** 2))
    return rmse(y_true, y_pred) / max(float(denom), 1e-15)


def parity_plot(y_true, y_pred, title: str, path: Path):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    pad = 0.05 * max(hi - lo, 1e-12)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=20, alpha=0.75)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad])
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference energy")
    ax.set_ylabel("Predicted energy")
    ax.set_title(
        f"{title}\n"
        f"RMSE={rmse(y_true, y_pred):.3e}, "
        f"rel={rel_rmse(y_true, y_pred):.3e}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def model_metrics(model, X_train, y_train, X_test, y_test):
    pred_train = model.predict(X_train)
    pred_test = None if X_test is None else model.predict(X_test)

    result = {
        "train_rmse": rmse(y_train, pred_train),
        "train_rel_rmse": rel_rmse(y_train, pred_train),
        "test_rmse": None,
        "test_rel_rmse": None,
        "pred_train": pred_train,
        "pred_test": pred_test,
    }

    if X_test is not None:
        result["test_rmse"] = rmse(y_test, pred_test)
        result["test_rel_rmse"] = rel_rmse(y_test, pred_test)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", default="random_rotations.xyz")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--test-every", type=int, default=5)

    parser.add_argument("--lmax", type=int, default=5)
    parser.add_argument("--nmax", type=int, default=5)
    parser.add_argument("--cutoff", type=float, default=4.5)
    parser.add_argument("--radial-gaussian-width", type=float, default=2.0)

    parser.add_argument(
        "--normalize-power-spectrum",
        action="store_true",
        help="Use AniSOAP's internal power-spectrum normalization.",
    )
    parser.add_argument(
        "--aggregate",
        choices=("mean", "sum"),
        default="mean",
        help=(
            "power_spectrum(mean_over_samples=True) returns system means. "
            "'sum' multiplies by atom count afterward."
        ),
    )
    parser.add_argument(
        "--alphas",
        default="1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/direct_anisoap_ridgecv",
    )

    args = parser.parse_args()

    xyz = Path(args.xyz).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, y = load_frames_and_energies(
        xyz,
        args.stride,
        args.max_frames,
    )

    calculator = EllipsoidalDensityProjection(
        max_angular=args.lmax,
        radial_basis_name="gto",
        cutoff_radius=args.cutoff,
        radial_gaussian_width=args.radial_gaussian_width,
        max_radial=args.nmax,
        rotation_key="quaternions",
        rotation_type="quaternion",
        dtype=torch.float64,
    )

    with torch.no_grad():
        X_torch = calculator.power_spectrum(
            frames=frames,
            mean_over_samples=True,
            show_progress=True,
            normalize=args.normalize_power_spectrum,
        )

    X = X_torch.detach().cpu().numpy()

    if X.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {X.shape}")

    if args.aggregate == "sum":
        counts = np.array([len(atoms) for atoms in frames], dtype=float)
        X = X * counts[:, None]

    train_idx, test_idx = split_indices(len(frames), args.test_every)

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_test = None
    y_test = None
    if len(test_idx):
        X_test = X[test_idx]
        y_test = y[test_idx]

    scaler_probe = StandardScaler()
    X_train_scaled = scaler_probe.fit_transform(X_train)
    A = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])

    singular_values = np.linalg.svd(A, compute_uv=False)
    tolerance = 1e-10 * singular_values.max()
    rank = int(np.sum(singular_values > tolerance))
    condition_number = float(
        singular_values.max() / max(singular_values.min(), 1e-300)
    )

    print("data")
    print("----")
    print("xyz:", xyz)
    print("n_frames:", len(frames))
    print("n_train:", len(train_idx))
    print("n_test:", len(test_idx))
    print("n_features:", X.shape[1])
    print("aggregate:", args.aggregate)
    print("power-spectrum normalized:", args.normalize_power_spectrum)
    print(
        "y_train min/max/mean/std:",
        float(y_train.min()),
        float(y_train.max()),
        float(y_train.mean()),
        float(y_train.std()),
    )

    print()
    print("conditioning")
    print("------------")
    print("A shape:", A.shape)
    print("rank:", rank)
    print("largest singular:", float(singular_values.max()))
    print("smallest singular:", float(singular_values.min()))
    print("condition number:", condition_number)

    alphas = np.array(
        [float(value) for value in args.alphas.split(",") if value.strip()]
    )

    models = {
        "linear_regression": make_pipeline(
            StandardScaler(),
            LinearRegression(),
        ),
    }

    for alpha in alphas:
        models[f"ridge_{alpha:.0e}"] = make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(alpha)),
        )

    models["ridgecv"] = make_pipeline(
        StandardScaler(),
        RidgeCV(
            alphas=alphas,
            scoring="neg_mean_squared_error",
        ),
    )

    summary = {
        "xyz": str(xyz),
        "stride": args.stride,
        "test_every": args.test_every,
        "lmax": args.lmax,
        "nmax": args.nmax,
        "cutoff": args.cutoff,
        "radial_gaussian_width": args.radial_gaussian_width,
        "aggregate": args.aggregate,
        "normalize_power_spectrum": args.normalize_power_spectrum,
        "n_frames": len(frames),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_features": int(X.shape[1]),
        "rank": rank,
        "condition_number": condition_number,
        "models": {},
    }

    print()
    print("fits")
    print("----")

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = model_metrics(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        extra = ""
        if name == "ridgecv":
            selected_alpha = model.named_steps["ridgecv"].alpha_
            extra = f" alpha={selected_alpha:.3e}"
            metrics["selected_alpha"] = float(selected_alpha)

        print(
            f"{name:>20s} "
            f"train_rmse={metrics['train_rmse']:.8e} "
            f"train_rel={metrics['train_rel_rmse']:.8e}"
            + (
                ""
                if X_test is None
                else (
                    f" test_rmse={metrics['test_rmse']:.8e} "
                    f"test_rel={metrics['test_rel_rmse']:.8e}"
                )
            )
            + extra
        )

        model_dir = output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        parity_plot(
            y_train,
            metrics["pred_train"],
            f"{name}: train",
            model_dir / "train_energy_parity.png",
        )

        if X_test is not None:
            parity_plot(
                y_test,
                metrics["pred_test"],
                f"{name}: test",
                model_dir / "test_energy_parity.png",
            )

        summary["models"][name] = {
            key: value
            for key, value in metrics.items()
            if key not in ("pred_train", "pred_test")
        }

    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print()
    print("wrote results to", output_dir)


if __name__ == "__main__":
    main()
