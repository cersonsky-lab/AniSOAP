from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from anisoap.representations.ellipsoidal_density_projection import (
        EllipsoidalDensityProjection,
    )
except Exception:
    from anisoap.representations.ellipsoidal_density_projection_tensor_core import (
        EllipsoidalDensityProjection,
    )


torch.set_default_dtype(torch.float64)


def frame_energy(atoms) -> float:
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        pass

    for key in ("energy", "Energy", "E", "potential_energy"):
        if key in atoms.info:
            return float(atoms.info[key])
        
    raise KeyError(
        "Could not find energy. Tried atoms.get_potential_energy() and "
        "info keys: energy, Energy, E, potential_energy."
    )


def read_frames(path: Path, *, stride: int, max_frames: int | None):
    frames = read(path, ":")[::stride]
    if max_frames is not None:
        frames = frames[:max_frames]

    for i, frame in enumerate(frames):
        if "energy" not in frame.info:
            frame.info['energy'] = frame.get_potential_energy()
        if "quaternions" not in frame.arrays:
            raise KeyError(f"frame {i} missing arrays['quaternions']")
        for key in ("c_diameter[1]", "c_diameter[2]", "c_diameter[3]"):
            if key not in frame.arrays:
                raise KeyError(f"frame {i} missing arrays[{key!r}]")

        q = np.asarray(frame.arrays["quaternions"], dtype=np.float64)
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        if np.any(q_norm == 0.0):
            raise ValueError(f"frame {i} has zero-norm quaternion")
        frame.arrays["quaternions"] = q / q_norm

        if not frame.pbc.any():
            frame.cell = [0.0, 0.0, 0.0]

    return frames


def train_test_indices(n: int, test_every: int | None):
    if test_every is None or test_every <= 0:
        return list(range(n)), []

    test = [i for i in range(n) if i % test_every == 0]
    train = [i for i in range(n) if i % test_every != 0]
    return train, test


def build_anisoap_power_spectrum(
    frames,
    *,
    lmax: int,
    nmax: int,
    cutoff: float,
    radial_gaussian_width: float,
    normalize: bool,
    aggregate: str,
):
    calculator = EllipsoidalDensityProjection(
        max_angular=lmax,
        radial_basis_name="gto",
        cutoff_radius=float(cutoff),
        radial_gaussian_width=float(radial_gaussian_width),
        max_radial=int(nmax),
        rotation_key="quaternions",
        rotation_type="quaternion",
        species=[0],
        dtype=torch.float64,
    )

    with torch.no_grad():
        X = calculator.power_spectrum(
            frames=frames,
            mean_over_samples=True,
            normalize=bool(normalize),
            show_progress=True,
        )

    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    else:
        X = np.asarray(X)

    X = np.asarray(X, dtype=np.float64)

    if aggregate == "sum":
        atom_counts = np.asarray([len(frame) for frame in frames], dtype=np.float64)
        X = X * atom_counts.reshape(-1, 1)
    elif aggregate == "mean":
        pass
    else:
        raise ValueError(f"unknown aggregate={aggregate!r}; use 'sum' or 'mean'")

    return X


def rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def rel_rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    denom = float(np.sqrt(np.mean((y_true - y_true.mean()) ** 2)))
    return rmse(y_true, y_pred) / max(denom, 1e-12)


def parity_plot(y_true, y_pred, *, title: str, output_path: Path):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=18, alpha=0.75)

    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    if hi > lo:
        pad = 0.05 * (hi - lo)
        lo -= pad
        hi += pad
        plt.plot([lo, hi], [lo, hi])
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel("reference energy")
    plt.ylabel("predicted energy")
    plt.title(
        f"{title}\n"
        f"RMSE={rmse(y_true, y_pred):.3e}, rel RMSE={rel_rmse(y_true, y_pred):.3e}"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
        .replace("=", "")
    )


def print_scores(name, model, X_train, y_train, X_test, y_test, output_dir: Path):
    pred_train = model.predict(X_train).reshape(-1)
    pred_test = model.predict(X_test).reshape(-1) if X_test is not None else None

    msg = (
        f"{name:>24s} "
        f"train_rmse={rmse(y_train, pred_train):.8e} "
        f"train_rel={rel_rmse(y_train, pred_train):.8e}"
    )
    if pred_test is not None:
        msg += (
            f" test_rmse={rmse(y_test, pred_test):.8e} "
            f"test_rel={rel_rmse(y_test, pred_test):.8e}"
        )

    if hasattr(model, "named_steps") and "ridgecv" in model.named_steps:
        msg += f" alpha={model.named_steps['ridgecv'].alpha_:.3e}"
    elif hasattr(model, "named_steps") and "ridge" in model.named_steps:
        msg += f" alpha={model.named_steps['ridge'].alpha:.3e}"

    print(msg)

    model_dir = output_dir / safe_name(name)
    parity_plot(
        y_train,
        pred_train,
        title=f"{name}: train",
        output_path=model_dir / "train_energy_parity.png",
    )
    if pred_test is not None:
        parity_plot(
            y_test,
            pred_test,
            title=f"{name}: test",
            output_path=model_dir / "test_energy_parity.png",
        )

    return {
        "train_rmse": rmse(y_train, pred_train),
        "train_rel_rmse": rel_rmse(y_train, pred_train),
        "test_rmse": None if pred_test is None else rmse(y_test, pred_test),
        "test_rel_rmse": None if pred_test is None else rel_rmse(y_test, pred_test),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", default="random_rotations.xyz")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--test-every", type=int, default=5)
    parser.add_argument("--lmax", type=int, default=5)
    parser.add_argument("--nmax", type=int, default=5)
    parser.add_argument("--cutoff", type=float, default=4.5)
    parser.add_argument("--radial-gaussian-width", type=float, default=1.5)
    parser.add_argument("--aggregate", choices=["sum", "mean"], default="sum")
    parser.add_argument("--no-anisoap-normalize", action="store_true")
    parser.add_argument(
        "--alphas",
        default="1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/anisoap_direct_power_spectrum_ridgecv",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = read_frames(
        Path(args.xyz).expanduser().resolve(),
        stride=args.stride,
        max_frames=args.max_frames,
    )
    y = np.asarray([frame_energy(frame) for frame in frames], dtype=np.float64)

    X = build_anisoap_power_spectrum(
        frames,
        lmax=args.lmax,
        nmax=args.nmax,
        cutoff=args.cutoff,
        radial_gaussian_width=args.radial_gaussian_width,
        normalize=not args.no_anisoap_normalize,
        aggregate=args.aggregate,
    )

    train_indices, test_indices = train_test_indices(len(frames), args.test_every)

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices] if test_indices else None
    y_test = y[test_indices] if test_indices else None

    alphas = np.asarray(
        [float(x) for x in args.alphas.split(",") if x.strip()],
        dtype=np.float64,
    )

    print("data")
    print("----")
    print("n_frames:", len(frames))
    print("n_train:", len(train_indices))
    print("n_test:", len(test_indices))
    print("n_features:", X.shape[1])
    print("aggregate:", args.aggregate)
    print("anisoap normalize:", not args.no_anisoap_normalize)
    print("y_train min/max/mean/std:", float(y_train.min()), float(y_train.max()), float(y_train.mean()), float(y_train.std()))
    if y_test is not None:
        print("y_test  min/max/mean/std:", float(y_test.min()), float(y_test.max()), float(y_test.mean()), float(y_test.std()))

    print()
    print("conditioning after sklearn-style StandardScaler")
    print("--------------------------------------------")
    Xs = StandardScaler().fit_transform(X_train)
    A = np.column_stack([np.ones(Xs.shape[0]), Xs])
    s = np.linalg.svd(A, compute_uv=False)
    rank = int((s > 1e-10 * s.max()).sum())
    print("A shape:", A.shape)
    print("rank:", rank)
    print("largest singular:", float(s.max()))
    print("smallest singular:", float(s.min()))
    print("condition number:", float(s.max() / max(s.min(), 1e-300)))

    print()
    print("fits")
    print("----")

    metrics = {}

    models = []

    models.append(
        (
            "sklearn_LinearRegression",
            make_pipeline(
                StandardScaler(),
                LinearRegression(),
            ),
        )
    )

    for alpha in alphas:
        models.append(
            (
                f"sklearn_Ridge_alpha_{alpha:.0e}",
                make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=float(alpha), fit_intercept=True),
                ),
            )
        )

    ridge_cv = make_pipeline(
        StandardScaler(),
        RidgeCV(
            alphas=alphas,
            fit_intercept=True,
            scoring="neg_mean_squared_error",
        ),
    )
    models.append(("sklearn_RidgeCV", ridge_cv))

    for name, model in models:
        model.fit(X_train, y_train)
        metrics[name] = print_scores(
            name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            output_dir,
        )

    summary = {
        "xyz": str(Path(args.xyz).expanduser().resolve()),
        "stride": args.stride,
        "max_frames": args.max_frames,
        "test_every": args.test_every,
        "lmax": args.lmax,
        "nmax": args.nmax,
        "cutoff": args.cutoff,
        "radial_gaussian_width": args.radial_gaussian_width,
        "aggregate": args.aggregate,
        "anisoap_normalize": not args.no_anisoap_normalize,
        "n_frames": len(frames),
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "n_features": int(X.shape[1]),
        "rank_standardized_with_bias": rank,
        "singular_values": {
            "largest": float(s.max()),
            "smallest": float(s.min()),
            "condition_number": float(s.max() / max(s.min(), 1e-300)),
        },
        "metrics": metrics,
    }

    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print()
    print("wrote parity plots and metrics to", output_dir)


if __name__ == "__main__":
    main()
