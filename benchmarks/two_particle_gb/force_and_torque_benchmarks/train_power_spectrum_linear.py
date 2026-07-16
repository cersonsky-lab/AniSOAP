from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from skmatter.preprocessing import StandardFlexibleScaler

from anisoap.representations import EllipsoidalDensityProjection


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)

    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)),
    }


def save_parity(
    target: np.ndarray,
    prediction: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    output: Path,
) -> None:
    target = np.asarray(target).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)

    low = float(min(target.min(), prediction.min()))
    high = float(max(target.max(), prediction.max()))
    padding = 0.04 * max(high - low, 1.0e-12)
    low -= padding
    high += padding

    fig, axis = plt.subplots(figsize=(6.5, 6.0))

    axis.scatter(
        target[train_indices],
        prediction[train_indices],
        s=18,
        alpha=0.55,
        label="train",
    )
    axis.scatter(
        target[test_indices],
        prediction[test_indices],
        s=30,
        alpha=0.85,
        marker="x",
        label="test",
    )

    axis.plot(
        [low, high],
        [low, high],
        linestyle="--",
        linewidth=1.2,
    )

    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Reference energy")
    axis.set_ylabel("Predicted energy")
    axis.set_title("Direct AniSOAP power-spectrum Ridge parity")
    axis.legend()
    axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Direct AniSOAP power-spectrum Ridge baseline matching the "
            "reference notebook."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("random_rotations_gb.xyz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("power_spectrum_linear_results"),
    )
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--max-angular", type=int, default=10)
    parser.add_argument("--max-radial", type=int, default=10)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--radial-width", type=float, default=2.0)
    parser.add_argument("--basis-rcond", type=float, default=1.0e-6)
    parser.add_argument("--basis-tol", type=float, default=1.0e-2)

    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=2)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--variance-threshold", type=float, default=1.0e-12)
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize AniSOAP power-spectrum rows.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frames = read(args.input, ":")

    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    if not frames:
        raise RuntimeError(f"No frames found in {args.input}")

    energies = np.asarray(
        [frame.get_potential_energy() for frame in frames],
        dtype=float,
    ).reshape(-1, 1)

    print(f"frames: {len(frames)}")
    print(
        "energy range/std:",
        float(energies.min()),
        float(energies.max()),
        float(energies.std()),
    )

    calculator = EllipsoidalDensityProjection(
        max_angular=args.max_angular,
        max_radial=args.max_radial,
        radial_basis_name="gto",
        rotation_type="quaternion",
        rotation_key="quaternions",
        cutoff_radius=args.cutoff,
        radial_gaussian_width=args.radial_width,
        basis_rcond=args.basis_rcond,
        basis_tol=args.basis_tol,
        subtract_center_contribution=False,
        dtype=torch.float64,
    )

    print("computing power spectrum ...")

    power_spectrum = np.asarray(
        calculator.power_spectrum(
            frames=frames,
            mean_over_samples=True,
            normalize=True,
            show_progress=True,
        ),
        dtype=float,
    )

    if power_spectrum.ndim != 2:
        raise RuntimeError(
            "Expected a two-dimensional feature matrix, got "
            f"{power_spectrum.shape}"
        )

    finite_mask = np.isfinite(power_spectrum).all(axis=0)
    variance = power_spectrum.var(axis=0)
    variance_mask = variance >= args.variance_threshold
    feature_mask = finite_mask & variance_mask

    features = power_spectrum[:, feature_mask]

    if features.shape[1] == 0:
        raise RuntimeError("No usable power-spectrum features remain")

    print("raw feature shape:", power_spectrum.shape)
    print("retained feature shape:", features.shape)

    # Match the notebook exactly:
    # global feature scaling rather than column-wise standardization.
    x_scaler = StandardFlexibleScaler(column_wise=False).fit(features)
    x = x_scaler.transform(features)

    y_scaler = StandardFlexibleScaler(column_wise=True).fit(energies)
    y = y_scaler.transform(energies).reshape(-1)

    all_indices = np.arange(len(frames))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=args.test_fraction,
        random_state=args.split_seed,
    )

    model = RidgeCV(
        cv=args.cv,
        alphas=np.logspace(-5, 0, 20),
        fit_intercept=False,
    )
    model.fit(x[train_indices], y[train_indices])

    prediction_scaled = model.predict(x).reshape(-1, 1)
    prediction = y_scaler.inverse_transform(
        prediction_scaled
    ).reshape(-1)

    target = energies.reshape(-1)

    results = {
        "configuration": {
            "input": str(args.input),
            "n_frames": len(frames),
            "n_raw_features": int(power_spectrum.shape[1]),
            "n_retained_features": int(features.shape[1]),
            "max_angular": args.max_angular,
            "max_radial": args.max_radial,
            "cutoff": args.cutoff,
            "radial_width": args.radial_width,
            "basis_rcond": args.basis_rcond,
            "basis_tol": args.basis_tol,
            "subtract_center_contribution": False,
            "mean_over_samples": True,
            "normalize": bool(args.normalize),
            "selected_alpha": float(model.alpha_),
        },
        "train": metrics(
            target[train_indices],
            prediction[train_indices],
        ),
        "test": metrics(
            target[test_indices],
            prediction[test_indices],
        ),
        "all": metrics(target, prediction),
    }

    np.savez(
        args.output / "predictions.npz",
        energy_target=target,
        energy_prediction=prediction,
        train_indices=train_indices,
        test_indices=test_indices,
        feature_mask=feature_mask,
        ridge_coefficients=np.asarray(model.coef_),
    )

    save_parity(
        target,
        prediction,
        train_indices,
        test_indices,
        args.output / "parity_energy.png",
    )

    with open(args.output / "metrics.json", "w") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Artifacts written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
