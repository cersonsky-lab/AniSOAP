from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.io import read
from scipy.spatial.transform import Rotation
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skmatter.preprocessing import StandardFlexibleScaler

from anisoap.representations import EllipsoidalDensityProjection


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit energies with a normalized AniSOAP power-spectrum Ridge "
            "model, then evaluate conservative forces and torques using "
            "central finite differences."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("random_rotations_gb.xyz"),
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        default=None,
        help="Explicit training trajectory for publication split mode.",
    )
    parser.add_argument(
        "--validation-input",
        type=Path,
        default=None,
        help="Explicit validation trajectory used only for alpha selection.",
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=None,
        help="Explicit untouched test trajectory used only for final metrics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gb_lr_finite_difference"),
    )
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--max-angular", type=int, default=10)
    parser.add_argument("--max-radial", type=int, default=10)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--radial-width", type=float, default=2.0)
    parser.add_argument("--basis-rcond", type=float, default=1.0e-6)
    parser.add_argument("--basis-tol", type=float, default=1.0e-2)

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Fixed Ridge alpha. When omitted, RidgeCV selects alpha from "
            "--alpha-min through --alpha-max."
        ),
    )
    parser.add_argument("--alpha-min", type=float, default=1.0e-10)
    parser.add_argument("--alpha-max", type=float, default=1.0e0)
    parser.add_argument("--alpha-count", type=int, default=61)

    parser.add_argument(
        "--position-step",
        type=float,
        default=1.0e-4,
        help="Central finite-difference position step in coordinate units.",
    )
    parser.add_argument(
        "--rotation-step",
        type=float,
        default=1.0e-4,
        help="Central finite-difference physical rotation step in radians.",
    )
    parser.add_argument(
        "--fd-batch-size",
        type=int,
        default=128,
        help="Maximum number of perturbed structures per AniSOAP call.",
    )

    parser.add_argument(
        "--quaternion-order",
        choices=("wxyz", "xyzw"),
        default="wxyz",
    )
    parser.add_argument(
        "--quaternion-matrix-direction",
        choices=("space_to_body", "body_to_space"),
        default="space_to_body",
    )
    parser.add_argument(
        "--torque-target-frame",
        choices=("body", "space"),
        default="body",
    )

    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=1.0e-14,
    )

    return parser.parse_args()


def metric_dictionary(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(target, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)

    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)),
        "target_std": float(np.std(target)),
        "prediction_std": float(np.std(prediction)),
    }


def quaternion_to_xyzw(
    quaternion: np.ndarray,
    order: str,
) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float)

    if quaternion.shape[-1] != 4:
        raise ValueError(
            f"Quaternion array must end in dimension 4, got {quaternion.shape}"
        )

    if order == "xyzw":
        xyzw = quaternion.copy()
    elif order == "wxyz":
        xyzw = quaternion[..., [1, 2, 3, 0]].copy()
    else:
        raise ValueError(f"Unsupported quaternion order {order!r}")

    norms = np.linalg.norm(xyzw, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-15):
        raise ValueError("Encountered zero-norm quaternion")

    return xyzw / norms


def xyzw_to_order(
    xyzw: np.ndarray,
    order: str,
) -> np.ndarray:
    xyzw = np.asarray(xyzw, dtype=float)

    if order == "xyzw":
        return xyzw.copy()
    if order == "wxyz":
        return xyzw[..., [3, 0, 1, 2]].copy()

    raise ValueError(f"Unsupported quaternion order {order!r}")


def stored_quaternions_to_space_to_body(
    quaternions: np.ndarray,
    *,
    order: str,
    matrix_direction: str,
) -> np.ndarray:
    xyzw = quaternion_to_xyzw(quaternions, order)
    matrices = Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix()
    matrices = matrices.reshape(quaternions.shape[:-1] + (3, 3))

    if matrix_direction == "space_to_body":
        return matrices
    if matrix_direction == "body_to_space":
        return np.swapaxes(matrices, -1, -2)

    raise ValueError(matrix_direction)


def space_to_body_to_stored_quaternions(
    matrices_sb: np.ndarray,
    *,
    order: str,
    matrix_direction: str,
) -> np.ndarray:
    matrices_sb = np.asarray(matrices_sb, dtype=float)

    if matrix_direction == "space_to_body":
        stored_matrices = matrices_sb
    elif matrix_direction == "body_to_space":
        stored_matrices = np.swapaxes(matrices_sb, -1, -2)
    else:
        raise ValueError(matrix_direction)

    xyzw = Rotation.from_matrix(
        stored_matrices.reshape(-1, 3, 3)
    ).as_quat()
    xyzw = xyzw.reshape(stored_matrices.shape[:-2] + (4,))

    return xyzw_to_order(xyzw, order)


def rotate_particle_in_space(
    frame: Atoms,
    particle: int,
    axis: int,
    angle: float,
    *,
    quaternion_order: str,
    quaternion_matrix_direction: str,
) -> Atoms:
    perturbed = frame.copy()

    quaternions = np.asarray(
        frame.arrays["quaternions"],
        dtype=float,
    ).copy()

    matrices_sb = stored_quaternions_to_space_to_body(
        quaternions,
        order=quaternion_order,
        matrix_direction=quaternion_matrix_direction,
    )

    rotation_vector = np.zeros(3, dtype=float)
    rotation_vector[axis] = angle
    active_space_rotation = Rotation.from_rotvec(
        rotation_vector
    ).as_matrix()

    # Let B map body coordinates to space coordinates and S = B^T.
    # A positive physical space rotation gives B' = R B, hence
    # S' = S R^T.
    matrices_sb[particle] = (
        matrices_sb[particle] @ active_space_rotation.T
    )

    perturbed.arrays["quaternions"] = (
        space_to_body_to_stored_quaternions(
            matrices_sb,
            order=quaternion_order,
            matrix_direction=quaternion_matrix_direction,
        )
    )

    return perturbed


def copy_with_position_displacement(
    frame: Atoms,
    particle: int,
    axis: int,
    displacement: float,
) -> Atoms:
    perturbed = frame.copy()
    perturbed.positions[particle, axis] += displacement
    return perturbed


def chunked(
    sequence: list[Atoms],
    size: int,
) -> Iterable[list[Atoms]]:
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


class RidgeEnergyModel:
    def __init__(
        self,
        calculator: EllipsoidalDensityProjection,
        feature_mask: np.ndarray,
        x_scaler: StandardFlexibleScaler,
        y_scaler: StandardFlexibleScaler,
        ridge: Ridge | RidgeCV,
        fd_batch_size: int,
    ) -> None:
        self.calculator = calculator
        self.feature_mask = np.asarray(feature_mask, dtype=bool)
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.ridge = ridge
        self.fd_batch_size = int(fd_batch_size)

    def raw_features(self, frames: list[Atoms]) -> np.ndarray:
        values = np.asarray(
            self.calculator.power_spectrum(
                frames=frames,
                mean_over_samples=True,
                normalize=True,
                show_progress=False,
            ),
            dtype=float,
        )

        if values.ndim != 2:
            raise RuntimeError(
                f"Expected feature matrix, got shape {values.shape}"
            )

        return values

    def predict(self, frames: list[Atoms]) -> np.ndarray:
        predictions = []

        for batch in chunked(frames, self.fd_batch_size):
            raw = self.raw_features(batch)
            selected = raw[:, self.feature_mask]
            scaled = self.x_scaler.transform(selected)

            prediction_scaled = np.asarray(
                self.ridge.predict(scaled),
                dtype=float,
            ).reshape(-1, 1)

            prediction = self.y_scaler.inverse_transform(
                prediction_scaled
            ).reshape(-1)

            predictions.append(prediction)

        return np.concatenate(predictions)


def finite_difference_forces(
    model: RidgeEnergyModel,
    frames: list[Atoms],
    step: float,
) -> np.ndarray:
    perturbed_frames: list[Atoms] = []
    metadata: list[tuple[int, int, int]] = []

    for frame_index, frame in enumerate(frames):
        for particle in range(len(frame)):
            for axis in range(3):
                perturbed_frames.append(
                    copy_with_position_displacement(
                        frame,
                        particle,
                        axis,
                        +step,
                    )
                )
                metadata.append((frame_index, particle, axis))

                perturbed_frames.append(
                    copy_with_position_displacement(
                        frame,
                        particle,
                        axis,
                        -step,
                    )
                )
                metadata.append((frame_index, particle, axis))

    energies = model.predict(perturbed_frames)

    forces = np.zeros((len(frames), 2, 3), dtype=float)

    for pair_index in range(0, len(energies), 2):
        frame_index, particle, axis = metadata[pair_index]

        energy_plus = energies[pair_index]
        energy_minus = energies[pair_index + 1]

        forces[frame_index, particle, axis] = -(
            energy_plus - energy_minus
        ) / (2.0 * step)

    return forces


def finite_difference_torques_space(
    model: RidgeEnergyModel,
    frames: list[Atoms],
    step: float,
    *,
    quaternion_order: str,
    quaternion_matrix_direction: str,
) -> np.ndarray:
    perturbed_frames: list[Atoms] = []
    metadata: list[tuple[int, int, int]] = []

    for frame_index, frame in enumerate(frames):
        for particle in range(len(frame)):
            for axis in range(3):
                perturbed_frames.append(
                    rotate_particle_in_space(
                        frame,
                        particle,
                        axis,
                        +step,
                        quaternion_order=quaternion_order,
                        quaternion_matrix_direction=(
                            quaternion_matrix_direction
                        ),
                    )
                )
                metadata.append((frame_index, particle, axis))

                perturbed_frames.append(
                    rotate_particle_in_space(
                        frame,
                        particle,
                        axis,
                        -step,
                        quaternion_order=quaternion_order,
                        quaternion_matrix_direction=(
                            quaternion_matrix_direction
                        ),
                    )
                )
                metadata.append((frame_index, particle, axis))

    energies = model.predict(perturbed_frames)

    torques = np.zeros((len(frames), 2, 3), dtype=float)

    for pair_index in range(0, len(energies), 2):
        frame_index, particle, axis = metadata[pair_index]

        energy_plus = energies[pair_index]
        energy_minus = energies[pair_index + 1]

        # The quaternion perturbation convention used above has positive
        # angle opposite to the generalized-coordinate convention of the
        # stored torque labels.
        torques[frame_index, particle, axis] = (
            energy_plus - energy_minus
        ) / (2.0 * step)

    return torques


def space_vectors_to_body(
    vectors_space: np.ndarray,
    frames: list[Atoms],
    *,
    quaternion_order: str,
    quaternion_matrix_direction: str,
) -> np.ndarray:
    converted = np.empty_like(vectors_space)

    for frame_index, frame in enumerate(frames):
        matrices_sb = stored_quaternions_to_space_to_body(
            np.asarray(frame.arrays["quaternions"], dtype=float),
            order=quaternion_order,
            matrix_direction=quaternion_matrix_direction,
        )

        converted[frame_index] = np.einsum(
            "aij,aj->ai",
            matrices_sb,
            vectors_space[frame_index],
        )

    return converted


def body_vectors_to_space(
    vectors_body: np.ndarray,
    frames: list[Atoms],
    *,
    quaternion_order: str,
    quaternion_matrix_direction: str,
) -> np.ndarray:
    converted = np.empty_like(vectors_body)

    for frame_index, frame in enumerate(frames):
        matrices_sb = stored_quaternions_to_space_to_body(
            np.asarray(frame.arrays["quaternions"], dtype=float),
            order=quaternion_order,
            matrix_direction=quaternion_matrix_direction,
        )

        converted[frame_index] = np.einsum(
            "aji,aj->ai",
            matrices_sb,
            vectors_body[frame_index],
        )

    return converted


def save_parity(
    target: np.ndarray,
    prediction: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    target_flat = np.asarray(target, dtype=float).reshape(-1)
    prediction_flat = np.asarray(prediction, dtype=float).reshape(-1)

    low = float(min(target_flat.min(), prediction_flat.min()))
    high = float(max(target_flat.max(), prediction_flat.max()))

    span = max(high - low, 1.0e-12)
    low -= 0.04 * span
    high += 0.04 * span

    values = metric_dictionary(target_flat, prediction_flat)

    figure, axis = plt.subplots(figsize=(6.3, 6.0))
    axis.scatter(
        target_flat,
        prediction_flat,
        s=14,
        alpha=0.55,
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
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(
        f"{title}\n"
        f"R²={values['r2']:.4f}, RMSE={values['rmse']:.4g}"
    )
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)



def run_explicit_split_mode(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)

    split_paths = (
        args.train_input,
        args.validation_input,
        args.test_input,
    )

    if not all(path is not None for path in split_paths):
        raise ValueError(
            "Explicit split mode requires --train-input, "
            "--validation-input, and --test-input together."
        )

    if args.max_frames is not None:
        raise ValueError(
            "--max-frames is not supported with explicit publication splits"
        )

    train_frames = read(args.train_input, ":")
    validation_frames = read(args.validation_input, ":")
    test_frames = read(args.test_input, ":")

    for label, frames, path in (
        ("training", train_frames, args.train_input),
        ("validation", validation_frames, args.validation_input),
        ("test", test_frames, args.test_input),
    ):
        if not frames:
            raise RuntimeError(f"No {label} frames read from {path}")
        if any(len(frame) != 2 for frame in frames):
            raise RuntimeError(
                f"{label.capitalize()} split contains a non-dimer frame"
            )
        if any("torques" not in frame.arrays for frame in frames):
            raise RuntimeError(
                f"{label.capitalize()} split contains a frame without torques"
            )

    train_energy = np.asarray(
        [frame.get_potential_energy() for frame in train_frames],
        dtype=float,
    )
    validation_energy = np.asarray(
        [frame.get_potential_energy() for frame in validation_frames],
        dtype=float,
    )
    test_energy = np.asarray(
        [frame.get_potential_energy() for frame in test_frames],
        dtype=float,
    )
    test_force = np.asarray(
        [frame.get_forces() for frame in test_frames],
        dtype=float,
    )
    test_torque_stored = np.asarray(
        [frame.arrays["torques"] for frame in test_frames],
        dtype=float,
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

    def raw_features(frames: list[Atoms]) -> np.ndarray:
        return np.asarray(
            calculator.power_spectrum(
                frames=frames,
                mean_over_samples=True,
                normalize=True,
                show_progress=False,
            ),
            dtype=float,
        )

    print(
        "explicit split sizes: "
        f"train={len(train_frames)}, "
        f"validation={len(validation_frames)}, "
        f"test={len(test_frames)}"
    )

    print("computing normalized training power spectrum ...")
    train_raw = raw_features(train_frames)

    # Feature selection is based on training data only.
    finite_mask = np.isfinite(train_raw).all(axis=0)
    variance_mask = (
        np.var(train_raw, axis=0)
        >= args.variance_threshold
    )
    feature_mask = finite_mask & variance_mask

    if not np.any(feature_mask):
        raise RuntimeError("No usable training features remain")

    print("computing normalized validation power spectrum ...")
    validation_raw = raw_features(validation_frames)

    train_selected = train_raw[:, feature_mask]
    validation_selected = validation_raw[:, feature_mask]

    if not np.isfinite(validation_selected).all():
        raise RuntimeError(
            "Validation features contain non-finite retained values"
        )

    # Both scalers are fitted exclusively on training data.
    x_scaler = StandardFlexibleScaler(
        column_wise=False
    ).fit(train_selected)
    y_scaler = StandardFlexibleScaler(
        column_wise=True
    ).fit(train_energy.reshape(-1, 1))

    train_x = x_scaler.transform(train_selected)
    validation_x = x_scaler.transform(validation_selected)
    train_y = y_scaler.transform(
        train_energy.reshape(-1, 1)
    ).reshape(-1)

    if args.alpha is None:
        alpha_values = np.logspace(
            np.log10(args.alpha_min),
            np.log10(args.alpha_max),
            args.alpha_count,
        )
    else:
        alpha_values = np.asarray([args.alpha], dtype=float)

    alpha_scan = []
    best_alpha = None
    best_score = np.inf

    for alpha in alpha_values:
        candidate = Ridge(
            alpha=float(alpha),
            fit_intercept=False,
        )
        candidate.fit(train_x, train_y)

        validation_scaled_prediction = np.asarray(
            candidate.predict(validation_x),
            dtype=float,
        ).reshape(-1, 1)
        validation_prediction = y_scaler.inverse_transform(
            validation_scaled_prediction
        ).reshape(-1)

        validation_mse = float(
            np.mean(
                (validation_prediction - validation_energy) ** 2
            )
        )
        validation_r2 = float(
            r2_score(
                validation_energy,
                validation_prediction,
            )
        )

        alpha_scan.append(
            {
                "alpha": float(alpha),
                "validation_energy_mse": validation_mse,
                "validation_energy_r2": validation_r2,
            }
        )

        print(
            f"alpha={alpha:.3e} "
            f"validation_E_MSE={validation_mse:.6g} "
            f"validation_E_R2={validation_r2:.6f}"
        )

        if validation_mse < best_score:
            best_score = validation_mse
            best_alpha = float(alpha)

    if best_alpha is None:
        raise RuntimeError("Alpha scan produced no result")

    print(f"selected alpha: {best_alpha:.12g}")

    # Refit on train + validation after alpha selection. Scaling remains
    # defined by the training split only.
    fit_frames = train_frames + validation_frames
    fit_energy = np.concatenate(
        [train_energy, validation_energy],
        axis=0,
    )
    fit_raw = np.concatenate(
        [train_raw, validation_raw],
        axis=0,
    )
    fit_x = x_scaler.transform(
        fit_raw[:, feature_mask]
    )
    fit_y = y_scaler.transform(
        fit_energy.reshape(-1, 1)
    ).reshape(-1)

    ridge = Ridge(
        alpha=best_alpha,
        fit_intercept=False,
    )
    ridge.fit(fit_x, fit_y)

    model = RidgeEnergyModel(
        calculator=calculator,
        feature_mask=feature_mask,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        ridge=ridge,
        fd_batch_size=args.fd_batch_size,
    )

    # The test split is first touched after model and alpha selection.
    print("predicting untouched test energies ...")
    test_energy_prediction = model.predict(test_frames)

    print("computing untouched test finite-difference forces ...")
    test_force_prediction = finite_difference_forces(
        model,
        test_frames,
        args.position_step,
    )

    print("computing untouched test finite-difference torques ...")
    test_torque_space_prediction = finite_difference_torques_space(
        model,
        test_frames,
        args.rotation_step,
        quaternion_order=args.quaternion_order,
        quaternion_matrix_direction=(
            args.quaternion_matrix_direction
        ),
    )

    test_torque_body_prediction = space_vectors_to_body(
        test_torque_space_prediction,
        test_frames,
        quaternion_order=args.quaternion_order,
        quaternion_matrix_direction=(
            args.quaternion_matrix_direction
        ),
    )

    if args.torque_target_frame == "body":
        test_torque_body = test_torque_stored
        test_torque_space = body_vectors_to_space(
            test_torque_body,
            test_frames,
            quaternion_order=args.quaternion_order,
            quaternion_matrix_direction=(
                args.quaternion_matrix_direction
            ),
        )
    else:
        test_torque_space = test_torque_stored
        test_torque_body = space_vectors_to_body(
            test_torque_space,
            test_frames,
            quaternion_order=args.quaternion_order,
            quaternion_matrix_direction=(
                args.quaternion_matrix_direction
            ),
        )

    results = {
        "configuration": {
            "train_input": str(args.train_input),
            "validation_input": str(args.validation_input),
            "test_input": str(args.test_input),
            "n_train": len(train_frames),
            "n_validation": len(validation_frames),
            "n_test": len(test_frames),
            "raw_feature_count": int(train_raw.shape[1]),
            "retained_feature_count": int(np.sum(feature_mask)),
            "coefficient_count": int(np.asarray(ridge.coef_).size),
            "max_angular": args.max_angular,
            "max_radial": args.max_radial,
            "cutoff": args.cutoff,
            "radial_width": args.radial_width,
            "basis_rcond": args.basis_rcond,
            "basis_tol": args.basis_tol,
            "normalize": True,
            "subtract_center_contribution": False,
            "selected_alpha": best_alpha,
            "position_step": args.position_step,
            "rotation_step": args.rotation_step,
            "quaternion_order": args.quaternion_order,
            "quaternion_matrix_direction": (
                args.quaternion_matrix_direction
            ),
            "torque_target_frame": args.torque_target_frame,
        },
        "test": {
            "energy": metric_dictionary(
                test_energy,
                test_energy_prediction,
            ),
            "force_components": metric_dictionary(
                test_force,
                test_force_prediction,
            ),
            "torque_components_body": metric_dictionary(
                test_torque_body,
                test_torque_body_prediction,
            ),
            "torque_components_space": metric_dictionary(
                test_torque_space,
                test_torque_space_prediction,
            ),
        },
        "alpha_scan": alpha_scan,
    }

    np.savez(
        args.output / "test_predictions.npz",
        energy_target=test_energy,
        energy_prediction=test_energy_prediction,
        force_target=test_force,
        force_prediction=test_force_prediction,
        torque_target_body=test_torque_body,
        torque_prediction_body=test_torque_body_prediction,
        torque_target_space=test_torque_space,
        torque_prediction_space=test_torque_space_prediction,
        feature_mask=feature_mask,
        ridge_coefficients=np.asarray(ridge.coef_),
    )

    with open(args.output / "metrics.json", "w") as handle:
        json.dump(results, handle, indent=2)

    save_parity(
        test_energy,
        test_energy_prediction,
        "Test: energy-only linear fit",
        "Reference energy",
        "Predicted energy",
        args.output / "test_parity_energy.png",
    )
    save_parity(
        test_force,
        test_force_prediction,
        "Test: finite-difference forces",
        "Reference force component",
        "Predicted force component",
        args.output / "test_parity_forces.png",
    )
    save_parity(
        test_torque_body,
        test_torque_body_prediction,
        "Test: finite-difference body torque",
        "Reference torque component",
        "Predicted torque component",
        args.output / "test_parity_torques_body.png",
    )
    save_parity(
        test_torque_space,
        test_torque_space_prediction,
        "Test: finite-difference space torque",
        "Reference torque component",
        "Predicted torque component",
        args.output / "test_parity_torques_space.png",
    )

    print(json.dumps(results["test"], indent=2))
    print(f"Artifacts written to {args.output.resolve()}")



def main() -> None:
    args = parse_arguments()

    explicit_split_requested = any(
        path is not None
        for path in (
            args.train_input,
            args.validation_input,
            args.test_input,
        )
    )
    if explicit_split_requested:
        run_explicit_split_mode(args)
        return

    if args.position_step <= 0.0:
        raise ValueError("--position-step must be positive")
    if args.rotation_step <= 0.0:
        raise ValueError("--rotation-step must be positive")
    if args.fd_batch_size <= 0:
        raise ValueError("--fd-batch-size must be positive")

    args.output.mkdir(parents=True, exist_ok=True)

    frames = read(args.input, ":")

    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    if not frames:
        raise RuntimeError(f"No frames read from {args.input}")

    if any(len(frame) != 2 for frame in frames):
        raise RuntimeError("This script expects two-particle configurations")

    energy_target = np.asarray(
        [frame.get_potential_energy() for frame in frames],
        dtype=float,
    )
    force_target = np.asarray(
        [frame.get_forces() for frame in frames],
        dtype=float,
    )

    if any("torques" not in frame.arrays for frame in frames):
        raise RuntimeError(
            "At least one frame has no per-atom 'torques' array"
        )

    torque_target_stored = np.asarray(
        [frame.arrays["torques"] for frame in frames],
        dtype=float,
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

    print(f"frames: {len(frames)}")
    print(
        "energy range/std:",
        float(energy_target.min()),
        float(energy_target.max()),
        float(energy_target.std()),
    )
    print("computing normalized AniSOAP power spectrum ...")

    raw_features = np.asarray(
        calculator.power_spectrum(
            frames=frames,
            mean_over_samples=True,
            normalize=True,
            show_progress=True,
        ),
        dtype=float,
    )

    finite_mask = np.isfinite(raw_features).all(axis=0)
    variance_mask = (
        np.var(raw_features, axis=0) >= args.variance_threshold
    )
    feature_mask = finite_mask & variance_mask
    features = raw_features[:, feature_mask]

    if features.shape[1] == 0:
        raise RuntimeError("No usable features remain")

    print("raw feature shape:", raw_features.shape)
    print("retained feature shape:", features.shape)

    # Match the successful notebook-style scaling.
    x_scaler = StandardFlexibleScaler(
        column_wise=False
    ).fit(features)
    y_scaler = StandardFlexibleScaler(
        column_wise=True
    ).fit(energy_target.reshape(-1, 1))

    x_scaled = x_scaler.transform(features)
    y_scaled = y_scaler.transform(
        energy_target.reshape(-1, 1)
    ).reshape(-1)

    if args.alpha is None:
        alphas = np.logspace(
            np.log10(args.alpha_min),
            np.log10(args.alpha_max),
            args.alpha_count,
        )

        ridge_cv = RidgeCV(
            alphas=alphas,
            fit_intercept=False,
            cv=None,
        )
        ridge_cv.fit(x_scaled, y_scaled)
        selected_alpha = float(ridge_cv.alpha_)

        # Refit cleanly at the selected alpha on all configurations.
        ridge: Ridge | RidgeCV = Ridge(
            alpha=selected_alpha,
            fit_intercept=False,
        )
        ridge.fit(x_scaled, y_scaled)
    else:
        selected_alpha = float(args.alpha)
        ridge = Ridge(
            alpha=selected_alpha,
            fit_intercept=False,
        )
        ridge.fit(x_scaled, y_scaled)

    model = RidgeEnergyModel(
        calculator=calculator,
        feature_mask=feature_mask,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        ridge=ridge,
        fd_batch_size=args.fd_batch_size,
    )

    energy_prediction = model.predict(frames)

    print(f"selected alpha: {selected_alpha:.12g}")
    print("computing finite-difference forces ...")

    force_prediction = finite_difference_forces(
        model,
        frames,
        args.position_step,
    )

    print("computing finite-difference torques ...")

    torque_prediction_space = finite_difference_torques_space(
        model,
        frames,
        args.rotation_step,
        quaternion_order=args.quaternion_order,
        quaternion_matrix_direction=(
            args.quaternion_matrix_direction
        ),
    )

    torque_prediction_body = space_vectors_to_body(
        torque_prediction_space,
        frames,
        quaternion_order=args.quaternion_order,
        quaternion_matrix_direction=(
            args.quaternion_matrix_direction
        ),
    )

    if args.torque_target_frame == "body":
        torque_target_body = torque_target_stored
        torque_target_space = body_vectors_to_space(
            torque_target_body,
            frames,
            quaternion_order=args.quaternion_order,
            quaternion_matrix_direction=(
                args.quaternion_matrix_direction
            ),
        )
    else:
        torque_target_space = torque_target_stored
        torque_target_body = space_vectors_to_body(
            torque_target_space,
            frames,
            quaternion_order=args.quaternion_order,
            quaternion_matrix_direction=(
                args.quaternion_matrix_direction
            ),
        )

    results = {
        "configuration": {
            "input": str(args.input),
            "n_frames": len(frames),
            "raw_feature_count": int(raw_features.shape[1]),
            "retained_feature_count": int(features.shape[1]),
            "max_angular": args.max_angular,
            "max_radial": args.max_radial,
            "cutoff": args.cutoff,
            "radial_width": args.radial_width,
            "basis_rcond": args.basis_rcond,
            "basis_tol": args.basis_tol,
            "normalize": True,
            "subtract_center_contribution": False,
            "selected_alpha": selected_alpha,
            "position_step": args.position_step,
            "rotation_step": args.rotation_step,
            "quaternion_order": args.quaternion_order,
            "quaternion_matrix_direction": (
                args.quaternion_matrix_direction
            ),
            "torque_target_frame": args.torque_target_frame,
        },
        "energy": metric_dictionary(
            energy_target,
            energy_prediction,
        ),
        "force_components": metric_dictionary(
            force_target,
            force_prediction,
        ),
        "torque_components_body": metric_dictionary(
            torque_target_body,
            torque_prediction_body,
        ),
        "torque_components_space": metric_dictionary(
            torque_target_space,
            torque_prediction_space,
        ),
    }

    np.savez(
        args.output / "predictions.npz",
        energy_target=energy_target,
        energy_prediction=energy_prediction,
        force_target=force_target,
        force_prediction=force_prediction,
        torque_target_body=torque_target_body,
        torque_prediction_body=torque_prediction_body,
        torque_target_space=torque_target_space,
        torque_prediction_space=torque_prediction_space,
        feature_mask=feature_mask,
        ridge_coefficients=np.asarray(ridge.coef_),
    )

    with open(args.output / "metrics.json", "w") as handle:
        json.dump(results, handle, indent=2)

    save_parity(
        energy_target,
        energy_prediction,
        "Energy parity",
        "Reference energy",
        "Predicted energy",
        args.output / "parity_energy.png",
    )
    save_parity(
        force_target,
        force_prediction,
        "Finite-difference force parity",
        "Reference force component",
        "Predicted force component",
        args.output / "parity_forces.png",
    )
    save_parity(
        torque_target_body,
        torque_prediction_body,
        "Finite-difference torque parity — body frame",
        "Reference torque component",
        "Predicted torque component",
        args.output / "parity_torques_body.png",
    )
    save_parity(
        torque_target_space,
        torque_prediction_space,
        "Finite-difference torque parity — space frame",
        "Reference torque component",
        "Predicted torque component",
        args.output / "parity_torques_space.png",
    )

    print(json.dumps(results, indent=2))
    print(f"Artifacts written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
