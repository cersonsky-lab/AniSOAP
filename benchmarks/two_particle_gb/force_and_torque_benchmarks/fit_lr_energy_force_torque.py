from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.io import read
from scipy.spatial.transform import Rotation
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from skmatter.preprocessing import StandardFlexibleScaler

from anisoap.representations import EllipsoidalDensityProjection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a conservative linear AniSOAP model to energies, forces, "
            "and optionally torques using finite-difference feature derivatives."
        )
    )

    parser.add_argument("--input", type=Path, default=Path("random_rotations_gb.xyz"))
    parser.add_argument(
        "--cache-output",
        type=Path,
        default=None,
        help=(
            "Precompute scaled AniSOAP features and finite-difference "
            "feature gradients for explicit train/validation/test splits, "
            "write them to this .npz file, and exit."
        ),
    )
    parser.add_argument(
        "--cache-input",
        type=Path,
        default=None,
        help=(
            "Load precomputed scaled AniSOAP features and feature gradients "
            "from this .npz file, then run the Ridge fit/alpha scan without "
            "recomputing descriptors."
        ),
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
    parser.add_argument("--output", type=Path, default=Path("gb_lr_eft"))
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--max-angular", type=int, default=10)
    parser.add_argument("--max-radial", type=int, default=10)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--radial-width", type=float, default=2.0)
    parser.add_argument("--basis-rcond", type=float, default=1.0e-6)
    parser.add_argument("--basis-tol", type=float, default=1.0e-2)
    parser.add_argument("--variance-threshold", type=float, default=1.0e-14)

    parser.add_argument("--position-step", type=float, default=1.0e-4)
    parser.add_argument("--rotation-step", type=float, default=1.0e-4)
    parser.add_argument("--feature-batch-size", type=int, default=128)

    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--force-weight", type=float, default=1.0)
    parser.add_argument("--torque-weight", type=float, default=0.0)

    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=2)

    parser.add_argument("--alpha-min", type=float, default=1.0e-12)
    parser.add_argument("--alpha-max", type=float, default=1.0e-2)
    parser.add_argument("--alpha-count", type=int, default=31)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Use a fixed alpha instead of scanning a validation grid.",
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
        "--torque-derivative-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help=(
            "Sign multiplying d(feature)/d(angle) when forming torque rows. "
            "Use +1 if the quaternion perturbation convention is opposite "
            "to the generalized torque coordinate; use -1 for tau=-dE/dtheta."
        ),
    )

    return parser.parse_args()


def chunks(items: list[Atoms], size: int) -> Iterable[list[Atoms]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def quaternion_to_xyzw(values: np.ndarray, order: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if order == "xyzw":
        result = values.copy()
    elif order == "wxyz":
        result = values[..., [1, 2, 3, 0]].copy()
    else:
        raise ValueError(order)

    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-15):
        raise ValueError("Encountered a zero-norm quaternion")

    return result / norms


def xyzw_to_order(values: np.ndarray, order: str) -> np.ndarray:
    if order == "xyzw":
        return np.asarray(values, dtype=float).copy()
    if order == "wxyz":
        return np.asarray(values, dtype=float)[..., [3, 0, 1, 2]].copy()
    raise ValueError(order)


def stored_to_space_to_body(
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


def space_to_body_to_stored(
    matrices_sb: np.ndarray,
    *,
    order: str,
    matrix_direction: str,
) -> np.ndarray:
    if matrix_direction == "space_to_body":
        stored = matrices_sb
    elif matrix_direction == "body_to_space":
        stored = np.swapaxes(matrices_sb, -1, -2)
    else:
        raise ValueError(matrix_direction)

    xyzw = Rotation.from_matrix(stored.reshape(-1, 3, 3)).as_quat()
    xyzw = xyzw.reshape(stored.shape[:-2] + (4,))
    return xyzw_to_order(xyzw, order)


def displaced_frame(
    frame: Atoms,
    particle: int,
    axis: int,
    displacement: float,
) -> Atoms:
    result = frame.copy()
    result.positions[particle, axis] += displacement
    return result


def rotated_frame(
    frame: Atoms,
    particle: int,
    axis: int,
    angle: float,
    *,
    quaternion_order: str,
    quaternion_matrix_direction: str,
) -> Atoms:
    result = frame.copy()

    quaternions = np.asarray(frame.arrays["quaternions"], dtype=float).copy()
    matrices_sb = stored_to_space_to_body(
        quaternions,
        order=quaternion_order,
        matrix_direction=quaternion_matrix_direction,
    )

    rotvec = np.zeros(3, dtype=float)
    rotvec[axis] = angle
    rotation_space = Rotation.from_rotvec(rotvec).as_matrix()

    # For body-to-space B and space-to-body S=B^T:
    # B' = R B, therefore S' = S R^T.
    matrices_sb[particle] = matrices_sb[particle] @ rotation_space.T

    result.arrays["quaternions"] = space_to_body_to_stored(
        matrices_sb,
        order=quaternion_order,
        matrix_direction=quaternion_matrix_direction,
    )

    return result


def body_to_space_vectors(
    vectors_body: np.ndarray,
    frames: list[Atoms],
    *,
    quaternion_order: str,
    quaternion_matrix_direction: str,
) -> np.ndarray:
    output = np.empty_like(vectors_body)

    for frame_index, frame in enumerate(frames):
        matrices_sb = stored_to_space_to_body(
            np.asarray(frame.arrays["quaternions"], dtype=float),
            order=quaternion_order,
            matrix_direction=quaternion_matrix_direction,
        )
        output[frame_index] = np.einsum(
            "aji,aj->ai",
            matrices_sb,
            vectors_body[frame_index],
        )

    return output


def space_to_body_vectors(
    vectors_space: np.ndarray,
    frames: list[Atoms],
    *,
    quaternion_order: str,
    quaternion_matrix_direction: str,
) -> np.ndarray:
    output = np.empty_like(vectors_space)

    for frame_index, frame in enumerate(frames):
        matrices_sb = stored_to_space_to_body(
            np.asarray(frame.arrays["quaternions"], dtype=float),
            order=quaternion_order,
            matrix_direction=quaternion_matrix_direction,
        )
        output[frame_index] = np.einsum(
            "aij,aj->ai",
            matrices_sb,
            vectors_space[frame_index],
        )

    return output


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)

    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)),
        "target_std": float(target.std()),
        "prediction_std": float(prediction.std()),
    }


def save_parity(
    target: np.ndarray,
    prediction: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    path: Path,
) -> None:
    target = np.asarray(target).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)

    low = float(min(target.min(), prediction.min()))
    high = float(max(target.max(), prediction.max()))
    span = max(high - low, 1.0e-12)
    low -= 0.04 * span
    high += 0.04 * span

    values = metrics(target, prediction)

    figure, axis = plt.subplots(figsize=(6.4, 6.0))
    axis.scatter(target, prediction, s=13, alpha=0.5)
    axis.plot([low, high], [low, high], linestyle="--", linewidth=1.2)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(
        f"{title}\n"
        f"R²={values['r2']:.4f}, RMSE={values['rmse']:.4g}"
    )
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


class FeatureComputer:
    def __init__(
        self,
        calculator: EllipsoidalDensityProjection,
        batch_size: int,
    ) -> None:
        self.calculator = calculator
        self.batch_size = batch_size

    def raw(self, frames: list[Atoms]) -> np.ndarray:
        result = []

        for batch in chunks(frames, self.batch_size):
            values = self.calculator.power_spectrum(
                frames=batch,
                mean_over_samples=True,
                normalize=True,
                show_progress=False,
            )
            result.append(np.asarray(values, dtype=float))

        return np.concatenate(result, axis=0)


def finite_difference_feature_derivatives(
    feature_computer: FeatureComputer,
    frames: list[Atoms],
    feature_mask: np.ndarray,
    x_scaler: StandardFlexibleScaler,
    *,
    position_step: float,
    rotation_step: float,
    quaternion_order: str,
    quaternion_matrix_direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_frames = len(frames)
    n_particles = len(frames[0])

    position_perturbations: list[Atoms] = []
    rotation_perturbations: list[Atoms] = []

    for frame in frames:
        for particle in range(n_particles):
            for axis in range(3):
                position_perturbations.append(
                    displaced_frame(frame, particle, axis, +position_step)
                )
                position_perturbations.append(
                    displaced_frame(frame, particle, axis, -position_step)
                )

                rotation_perturbations.append(
                    rotated_frame(
                        frame,
                        particle,
                        axis,
                        +rotation_step,
                        quaternion_order=quaternion_order,
                        quaternion_matrix_direction=quaternion_matrix_direction,
                    )
                )
                rotation_perturbations.append(
                    rotated_frame(
                        frame,
                        particle,
                        axis,
                        -rotation_step,
                        quaternion_order=quaternion_order,
                        quaternion_matrix_direction=quaternion_matrix_direction,
                    )
                )

    print(
        f"computing {len(position_perturbations)} position-perturbed feature rows ..."
    )
    position_raw = feature_computer.raw(position_perturbations)
    position_scaled = x_scaler.transform(position_raw[:, feature_mask])

    print(
        f"computing {len(rotation_perturbations)} rotation-perturbed feature rows ..."
    )
    rotation_raw = feature_computer.raw(rotation_perturbations)
    rotation_scaled = x_scaler.transform(rotation_raw[:, feature_mask])

    n_features = position_scaled.shape[1]

    position_pairs = position_scaled.reshape(
        n_frames,
        n_particles,
        3,
        2,
        n_features,
    )
    rotation_pairs = rotation_scaled.reshape(
        n_frames,
        n_particles,
        3,
        2,
        n_features,
    )

    dfeatures_dr = (
        position_pairs[:, :, :, 0, :]
        - position_pairs[:, :, :, 1, :]
    ) / (2.0 * position_step)

    dfeatures_dtheta = (
        rotation_pairs[:, :, :, 0, :]
        - rotation_pairs[:, :, :, 1, :]
    ) / (2.0 * rotation_step)

    return dfeatures_dr, dfeatures_dtheta


def make_augmented_system(
    frame_indices: np.ndarray,
    x: np.ndarray,
    energy: np.ndarray,
    force: np.ndarray,
    torque_space: np.ndarray,
    dfeatures_dr: np.ndarray,
    dfeatures_dtheta: np.ndarray,
    *,
    energy_scale: float,
    force_scale: float,
    torque_scale: float,
    energy_weight: float,
    force_weight: float,
    torque_weight: float,
    torque_derivative_sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrices = []
    targets = []

    if energy_weight > 0.0:
        matrices.append(
            np.sqrt(energy_weight)
            * x[frame_indices]
            / energy_scale
        )
        targets.append(
            np.sqrt(energy_weight)
            * energy[frame_indices]
            / energy_scale
        )

    if force_weight > 0.0:
        force_design = -dfeatures_dr[frame_indices].reshape(
            -1,
            x.shape[1],
        )
        force_target = force[frame_indices].reshape(-1)

        matrices.append(
            np.sqrt(force_weight)
            * force_design
            / force_scale
        )
        targets.append(
            np.sqrt(force_weight)
            * force_target
            / force_scale
        )

    if torque_weight > 0.0:
        torque_design = (
            torque_derivative_sign
            * dfeatures_dtheta[frame_indices].reshape(
                -1,
                x.shape[1],
            )
        )
        torque_target = torque_space[frame_indices].reshape(-1)

        matrices.append(
            np.sqrt(torque_weight)
            * torque_design
            / torque_scale
        )
        targets.append(
            np.sqrt(torque_weight)
            * torque_target
            / torque_scale
        )

    if not matrices:
        raise ValueError("At least one E/F/T weight must be positive")

    return np.concatenate(matrices, axis=0), np.concatenate(targets)


def predictions_from_coefficients(
    coefficients: np.ndarray,
    x: np.ndarray,
    dfeatures_dr: np.ndarray,
    dfeatures_dtheta: np.ndarray,
    *,
    torque_derivative_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy = x @ coefficients

    force = -np.einsum(
        "napf,f->nap",
        dfeatures_dr,
        coefficients,
    )

    torque_space = torque_derivative_sign * np.einsum(
        "napf,f->nap",
        dfeatures_dtheta,
        coefficients,
    )

    return energy, force, torque_space


def normalized_validation_score(
    energy_target: np.ndarray,
    energy_prediction: np.ndarray,
    force_target: np.ndarray,
    force_prediction: np.ndarray,
    torque_target: np.ndarray,
    torque_prediction: np.ndarray,
    indices: np.ndarray,
    *,
    energy_scale: float,
    force_scale: float,
    torque_scale: float,
    energy_weight: float,
    force_weight: float,
    torque_weight: float,
) -> float:
    score = 0.0
    total_weight = 0.0

    if energy_weight > 0.0:
        error = (
            energy_prediction[indices]
            - energy_target[indices]
        ) / energy_scale
        score += energy_weight * float(np.mean(error**2))
        total_weight += energy_weight

    if force_weight > 0.0:
        error = (
            force_prediction[indices]
            - force_target[indices]
        ) / force_scale
        score += force_weight * float(np.mean(error**2))
        total_weight += force_weight

    if torque_weight > 0.0:
        error = (
            torque_prediction[indices]
            - torque_target[indices]
        ) / torque_scale
        score += torque_weight * float(np.mean(error**2))
        total_weight += torque_weight

    return score / total_weight



def _read_required_frames(path: Path, label: str) -> list[Atoms]:
    frames = read(path, ":")

    if not frames:
        raise RuntimeError(f"No {label} frames found in {path}")

    return frames


def _extract_targets(
    frames: list[Atoms],
    *,
    quaternion_order: str,
    quaternion_matrix_direction: str,
    torque_target_frame: str,
) -> dict[str, np.ndarray]:
    energy = np.asarray(
        [frame.get_potential_energy() for frame in frames],
        dtype=float,
    )
    force = np.asarray(
        [frame.get_forces() for frame in frames],
        dtype=float,
    )

    if any("torques" not in frame.arrays for frame in frames):
        raise RuntimeError("Every frame must contain a 'torques' array")

    torque_stored = np.asarray(
        [frame.arrays["torques"] for frame in frames],
        dtype=float,
    )

    if torque_target_frame == "body":
        torque_body = torque_stored
        torque_space = body_to_space_vectors(
            torque_body,
            frames,
            quaternion_order=quaternion_order,
            quaternion_matrix_direction=quaternion_matrix_direction,
        )
    else:
        torque_space = torque_stored
        torque_body = space_to_body_vectors(
            torque_space,
            frames,
            quaternion_order=quaternion_order,
            quaternion_matrix_direction=quaternion_matrix_direction,
        )

    return {
        "energy": energy,
        "force": force,
        "torque_body": torque_body,
        "torque_space": torque_space,
    }


def _predict_split(
    coefficients: np.ndarray,
    x: np.ndarray,
    dfeatures_dr: np.ndarray,
    dfeatures_dtheta: np.ndarray,
    *,
    energy_mean: float,
    torque_derivative_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy_centered, force, torque_space = predictions_from_coefficients(
        coefficients,
        x,
        dfeatures_dr,
        dfeatures_dtheta,
        torque_derivative_sign=torque_derivative_sign,
    )

    return energy_centered + energy_mean, force, torque_space


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

    train_frames = _read_required_frames(
        args.train_input,
        "training",
    )
    validation_frames = _read_required_frames(
        args.validation_input,
        "validation",
    )
    test_frames = _read_required_frames(
        args.test_input,
        "test",
    )

    if args.max_frames is not None:
        raise ValueError(
            "--max-frames is not supported with explicit publication splits"
        )

    all_frame_groups = (
        train_frames,
        validation_frames,
        test_frames,
    )

    n_particles = len(train_frames[0])

    for frames in all_frame_groups:
        if any(len(frame) != n_particles for frame in frames):
            raise RuntimeError(
                "All train, validation, and test frames must have the "
                "same particle count"
            )

    train_targets = _extract_targets(
        train_frames,
        quaternion_order=args.quaternion_order,
        quaternion_matrix_direction=args.quaternion_matrix_direction,
        torque_target_frame=args.torque_target_frame,
    )
    validation_targets = _extract_targets(
        validation_frames,
        quaternion_order=args.quaternion_order,
        quaternion_matrix_direction=args.quaternion_matrix_direction,
        torque_target_frame=args.torque_target_frame,
    )
    test_targets = _extract_targets(
        test_frames,
        quaternion_order=args.quaternion_order,
        quaternion_matrix_direction=args.quaternion_matrix_direction,
        torque_target_frame=args.torque_target_frame,
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

    feature_computer = FeatureComputer(
        calculator,
        args.feature_batch_size,
    )

    print(
        "explicit split sizes: "
        f"train={len(train_frames)}, "
        f"validation={len(validation_frames)}, "
        f"test={len(test_frames)}"
    )

    print("computing normalized training AniSOAP features ...")
    train_raw = feature_computer.raw(train_frames)

    # Feature selection is based only on training data.
    finite_mask = np.isfinite(train_raw).all(axis=0)
    variance_mask = (
        np.var(train_raw, axis=0)
        >= args.variance_threshold
    )
    feature_mask = finite_mask & variance_mask

    if not np.any(feature_mask):
        raise RuntimeError("No usable training features remain")

    train_selected = train_raw[:, feature_mask]

    print("computing normalized validation AniSOAP features ...")
    validation_raw = feature_computer.raw(validation_frames)
    validation_selected = validation_raw[:, feature_mask]

    print("computing normalized test AniSOAP features ...")
    test_raw = feature_computer.raw(test_frames)
    test_selected = test_raw[:, feature_mask]

    if not np.isfinite(validation_selected).all():
        raise RuntimeError(
            "Validation features contain non-finite values in retained columns"
        )
    if not np.isfinite(test_selected).all():
        raise RuntimeError(
            "Test features contain non-finite values in retained columns"
        )

    # The scaler is fitted only on training features.
    x_scaler = StandardFlexibleScaler(
        column_wise=False
    ).fit(train_selected)

    train_x = x_scaler.transform(train_selected)
    validation_x = x_scaler.transform(validation_selected)
    test_x = x_scaler.transform(test_selected)

    # All centering and observable scales are training-only quantities.
    energy_mean = float(np.mean(train_targets["energy"]))
    energy_scale = max(
        float(np.std(train_targets["energy"] - energy_mean)),
        1.0e-15,
    )
    force_scale = max(
        float(np.std(train_targets["force"])),
        1.0e-15,
    )
    torque_scale = max(
        float(np.std(train_targets["torque_space"])),
        1.0e-15,
    )

    train_energy_centered = (
        train_targets["energy"] - energy_mean
    )
    validation_energy_centered = (
        validation_targets["energy"] - energy_mean
    )

    print("computing training feature derivatives ...")
    train_dr, train_dtheta = finite_difference_feature_derivatives(
        feature_computer,
        train_frames,
        feature_mask,
        x_scaler,
        position_step=args.position_step,
        rotation_step=args.rotation_step,
        quaternion_order=args.quaternion_order,
        quaternion_matrix_direction=args.quaternion_matrix_direction,
    )

    print("computing validation feature derivatives ...")
    validation_dr, validation_dtheta = (
        finite_difference_feature_derivatives(
            feature_computer,
            validation_frames,
            feature_mask,
            x_scaler,
            position_step=args.position_step,
            rotation_step=args.rotation_step,
            quaternion_order=args.quaternion_order,
            quaternion_matrix_direction=(
                args.quaternion_matrix_direction
            ),
        )
    )

    train_indices = np.arange(len(train_frames))
    validation_indices = np.arange(len(validation_frames))

    train_matrix, train_target = make_augmented_system(
        train_indices,
        train_x,
        train_energy_centered,
        train_targets["force"],
        train_targets["torque_space"],
        train_dr,
        train_dtheta,
        energy_scale=energy_scale,
        force_scale=force_scale,
        torque_scale=torque_scale,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        torque_weight=args.torque_weight,
        torque_derivative_sign=args.torque_derivative_sign,
    )

    if args.alpha is None:
        alpha_values = np.logspace(
            np.log10(args.alpha_min),
            np.log10(args.alpha_max),
            args.alpha_count,
        )
    else:
        alpha_values = np.asarray([args.alpha], dtype=float)

    scan_rows = []
    best_score = np.inf
    best_alpha = None

    for alpha in alpha_values:
        model = Ridge(
            alpha=float(alpha),
            fit_intercept=False,
            solver="lsqr",
            tol=1.0e-10,
            max_iter=10000,
        )
        model.fit(train_matrix, train_target)

        coefficients = np.asarray(model.coef_, dtype=float)

        validation_energy, validation_force, validation_torque_space = (
            _predict_split(
                coefficients,
                validation_x,
                validation_dr,
                validation_dtheta,
                energy_mean=energy_mean,
                torque_derivative_sign=args.torque_derivative_sign,
            )
        )

        score = normalized_validation_score(
            validation_targets["energy"],
            validation_energy,
            validation_targets["force"],
            validation_force,
            validation_targets["torque_space"],
            validation_torque_space,
            validation_indices,
            energy_scale=energy_scale,
            force_scale=force_scale,
            torque_scale=torque_scale,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            torque_weight=args.torque_weight,
        )

        row = {
            "alpha": float(alpha),
            "validation_score": float(score),
            "validation_energy_r2": float(
                r2_score(
                    validation_targets["energy"],
                    validation_energy,
                )
            ),
            "validation_force_r2": float(
                r2_score(
                    validation_targets["force"].reshape(-1),
                    validation_force.reshape(-1),
                )
            ),
            "validation_torque_r2": float(
                r2_score(
                    validation_targets["torque_space"].reshape(-1),
                    validation_torque_space.reshape(-1),
                )
            ),
        }
        scan_rows.append(row)

        print(
            f"alpha={alpha:.3e} "
            f"score={score:.6g} "
            f"E_R2={row['validation_energy_r2']:.5f} "
            f"F_R2={row['validation_force_r2']:.5f} "
            f"T_R2={row['validation_torque_r2']:.5f}"
        )

        if score < best_score:
            best_score = score
            best_alpha = float(alpha)

    if best_alpha is None:
        raise RuntimeError("Alpha scan produced no result")

    print(f"selected alpha: {best_alpha:.12g}")

    # Refit on train + validation after alpha selection. The feature mask,
    # feature scaler, target center, and target scales remain training-only.
    fit_x = np.concatenate(
        [train_x, validation_x],
        axis=0,
    )
    fit_energy_centered = np.concatenate(
        [train_energy_centered, validation_energy_centered],
        axis=0,
    )
    fit_force = np.concatenate(
        [train_targets["force"], validation_targets["force"]],
        axis=0,
    )
    fit_torque_space = np.concatenate(
        [
            train_targets["torque_space"],
            validation_targets["torque_space"],
        ],
        axis=0,
    )
    fit_dr = np.concatenate(
        [train_dr, validation_dr],
        axis=0,
    )
    fit_dtheta = np.concatenate(
        [train_dtheta, validation_dtheta],
        axis=0,
    )

    fit_indices = np.arange(len(fit_x))

    fit_matrix, fit_target = make_augmented_system(
        fit_indices,
        fit_x,
        fit_energy_centered,
        fit_force,
        fit_torque_space,
        fit_dr,
        fit_dtheta,
        energy_scale=energy_scale,
        force_scale=force_scale,
        torque_scale=torque_scale,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        torque_weight=args.torque_weight,
        torque_derivative_sign=args.torque_derivative_sign,
    )

    final_model = Ridge(
        alpha=best_alpha,
        fit_intercept=False,
        solver="lsqr",
        tol=1.0e-10,
        max_iter=10000,
    )
    final_model.fit(fit_matrix, fit_target)
    coefficients = np.asarray(final_model.coef_, dtype=float)

    # Test derivatives are computed only after model selection is complete.
    print("computing untouched test feature derivatives ...")
    test_dr, test_dtheta = finite_difference_feature_derivatives(
        feature_computer,
        test_frames,
        feature_mask,
        x_scaler,
        position_step=args.position_step,
        rotation_step=args.rotation_step,
        quaternion_order=args.quaternion_order,
        quaternion_matrix_direction=args.quaternion_matrix_direction,
    )

    test_energy, test_force, test_torque_space = _predict_split(
        coefficients,
        test_x,
        test_dr,
        test_dtheta,
        energy_mean=energy_mean,
        torque_derivative_sign=args.torque_derivative_sign,
    )

    test_torque_body = space_to_body_vectors(
        test_torque_space,
        test_frames,
        quaternion_order=args.quaternion_order,
        quaternion_matrix_direction=args.quaternion_matrix_direction,
    )

    results = {
        "configuration": {
            "train_input": str(args.train_input),
            "validation_input": str(args.validation_input),
            "test_input": str(args.test_input),
            "n_train": len(train_frames),
            "n_validation": len(validation_frames),
            "n_test": len(test_frames),
            "n_particles": n_particles,
            "raw_feature_count": int(train_raw.shape[1]),
            "retained_feature_count": int(np.sum(feature_mask)),
            "coefficient_count": int(coefficients.size),
            "normalize": True,
            "subtract_center_contribution": False,
            "max_angular": args.max_angular,
            "max_radial": args.max_radial,
            "cutoff": args.cutoff,
            "radial_width": args.radial_width,
            "basis_rcond": args.basis_rcond,
            "basis_tol": args.basis_tol,
            "position_step": args.position_step,
            "rotation_step": args.rotation_step,
            "energy_weight": args.energy_weight,
            "force_weight": args.force_weight,
            "torque_weight": args.torque_weight,
            "torque_derivative_sign": args.torque_derivative_sign,
            "selected_alpha": best_alpha,
            "energy_mean_from_train": energy_mean,
            "energy_scale_from_train": energy_scale,
            "force_scale_from_train": force_scale,
            "torque_scale_from_train": torque_scale,
        },
        "test": {
            "energy": metrics(
                test_targets["energy"],
                test_energy,
            ),
            "force_components": metrics(
                test_targets["force"],
                test_force,
            ),
            "torque_components_space": metrics(
                test_targets["torque_space"],
                test_torque_space,
            ),
            "torque_components_body": metrics(
                test_targets["torque_body"],
                test_torque_body,
            ),
        },
        "alpha_scan": scan_rows,
    }

    np.savez(
        args.output / "test_predictions.npz",
        energy_target=test_targets["energy"],
        energy_prediction=test_energy,
        force_target=test_targets["force"],
        force_prediction=test_force,
        torque_target_space=test_targets["torque_space"],
        torque_prediction_space=test_torque_space,
        torque_target_body=test_targets["torque_body"],
        torque_prediction_body=test_torque_body,
        coefficients=coefficients,
        feature_mask=feature_mask,
    )

    with open(args.output / "metrics.json", "w") as handle:
        json.dump(results, handle, indent=2)

    save_parity(
        test_targets["energy"],
        test_energy,
        "Test: conservative linear fit — energy",
        "Reference energy",
        "Predicted energy",
        args.output / "test_parity_energy.png",
    )
    save_parity(
        test_targets["force"],
        test_force,
        "Test: conservative linear fit — forces",
        "Reference force component",
        "Predicted force component",
        args.output / "test_parity_forces.png",
    )
    save_parity(
        test_targets["torque_body"],
        test_torque_body,
        "Test: conservative linear fit — body torque",
        "Reference torque component",
        "Predicted torque component",
        args.output / "test_parity_torques_body.png",
    )
    save_parity(
        test_targets["torque_space"],
        test_torque_space,
        "Test: conservative linear fit — space torque",
        "Reference torque component",
        "Predicted torque component",
        args.output / "test_parity_torques_space.png",
    )

    print(json.dumps(results["test"], indent=2))
    print(f"Artifacts written to {args.output.resolve()}")




def _cache_read_frames(path: Path, label: str) -> list[Atoms]:
    frames = read(path, ":")

    if not frames:
        raise RuntimeError(f"No {label} frames read from {path}")

    return frames


def _cache_validate_frame_groups(
    frame_groups: dict[str, list[Atoms]],
) -> int:
    first_group = next(iter(frame_groups.values()))
    n_particles = len(first_group[0])

    # In ASE extended XYZ files, forces may be loaded as calculator
    # results rather than as frame.arrays["forces"].  The fitter uses
    # frame.get_forces(), so do not require a literal "forces" array here.
    required_arrays = {
        "quaternions",
        "torques",
        "c_diameter[1]",
        "c_diameter[2]",
        "c_diameter[3]",
    }

    for label, frames in frame_groups.items():
        for frame_index, frame in enumerate(frames):
            if len(frame) != n_particles:
                raise RuntimeError(
                    f"{label} frame {frame_index} has {len(frame)} "
                    f"particles; expected {n_particles}"
                )

            missing = required_arrays - set(frame.arrays)
            if missing:
                raise RuntimeError(
                    f"{label} frame {frame_index} missing arrays "
                    f"{sorted(missing)}"
                )

            try:
                forces = frame.get_forces()
            except Exception as exc:
                raise RuntimeError(
                    f"{label} frame {frame_index} does not provide forces "
                    "through frame.get_forces()"
                ) from exc

            forces = np.asarray(forces, dtype=float)
            if forces.shape != (n_particles, 3):
                raise RuntimeError(
                    f"{label} frame {frame_index} has force shape "
                    f"{forces.shape}; expected {(n_particles, 3)}"
                )

    return n_particles


def _cache_extract_targets(
    frames: list[Atoms],
    *,
    quaternion_order: str,
    quaternion_matrix_direction: str,
    torque_target_frame: str,
) -> dict[str, np.ndarray]:
    energy = np.asarray(
        [frame.get_potential_energy() for frame in frames],
        dtype=float,
    )
    force = np.asarray(
        [frame.get_forces() for frame in frames],
        dtype=float,
    )
    torque_stored = np.asarray(
        [frame.arrays["torques"] for frame in frames],
        dtype=float,
    )

    if torque_target_frame == "body":
        torque_body = torque_stored
        torque_space = body_to_space_vectors(
            torque_body,
            frames,
            quaternion_order=quaternion_order,
            quaternion_matrix_direction=quaternion_matrix_direction,
        )
    elif torque_target_frame == "space":
        torque_space = torque_stored
        torque_body = space_to_body_vectors(
            torque_space,
            frames,
            quaternion_order=quaternion_order,
            quaternion_matrix_direction=quaternion_matrix_direction,
        )
    else:
        raise ValueError(torque_target_frame)

    matrices_space_to_body = np.asarray(
        [
            stored_to_space_to_body(
                np.asarray(frame.arrays["quaternions"], dtype=float),
                order=quaternion_order,
                matrix_direction=quaternion_matrix_direction,
            )
            for frame in frames
        ],
        dtype=float,
    )

    return {
        "energy": energy,
        "force": force,
        "torque_body": torque_body,
        "torque_space": torque_space,
        "matrices_space_to_body": matrices_space_to_body,
    }


def run_cache_build_mode(args: argparse.Namespace) -> None:
    if args.cache_output is None:
        raise ValueError("--cache-output is required for cache build mode")

    if not (
        args.train_input is not None
        and args.validation_input is not None
        and args.test_input is not None
    ):
        raise ValueError(
            "Cache build mode requires --train-input, "
            "--validation-input, and --test-input"
        )

    if args.max_frames is not None:
        raise ValueError("--max-frames is not supported in cache build mode")

    args.cache_output.parent.mkdir(parents=True, exist_ok=True)

    frame_groups = {
        "train": _cache_read_frames(args.train_input, "training"),
        "validation": _cache_read_frames(
            args.validation_input,
            "validation",
        ),
        "test": _cache_read_frames(args.test_input, "test"),
    }

    n_particles = _cache_validate_frame_groups(frame_groups)

    target_groups = {
        name: _cache_extract_targets(
            frames,
            quaternion_order=args.quaternion_order,
            quaternion_matrix_direction=args.quaternion_matrix_direction,
            torque_target_frame=args.torque_target_frame,
        )
        for name, frames in frame_groups.items()
    }

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

    feature_computer = FeatureComputer(
        calculator,
        args.feature_batch_size,
    )

    print(
        "cache split sizes: "
        f"train={len(frame_groups['train'])}, "
        f"validation={len(frame_groups['validation'])}, "
        f"test={len(frame_groups['test'])}",
        flush=True,
    )

    raw_features = {}
    for split in ("train", "validation", "test"):
        print(
            f"computing normalized {split} AniSOAP features ...",
            flush=True,
        )
        raw_features[split] = feature_computer.raw(frame_groups[split])

    # Feature selection is fitted on training data only.
    finite_mask = np.isfinite(raw_features["train"]).all(axis=0)
    variance_mask = (
        np.var(raw_features["train"], axis=0)
        >= args.variance_threshold
    )
    feature_mask = finite_mask & variance_mask

    if not np.any(feature_mask):
        raise RuntimeError("No usable training features remain")

    print("raw feature shape:", raw_features["train"].shape, flush=True)
    print("retained feature count:", int(np.sum(feature_mask)), flush=True)

    selected_features = {
        split: raw_features[split][:, feature_mask]
        for split in ("train", "validation", "test")
    }

    for split in ("validation", "test"):
        if not np.isfinite(selected_features[split]).all():
            raise RuntimeError(
                f"{split} features contain non-finite values in retained "
                "columns"
            )

    x_scaler = StandardFlexibleScaler(
        column_wise=False
    ).fit(selected_features["train"])

    x = {
        split: x_scaler.transform(selected_features[split])
        for split in ("train", "validation", "test")
    }

    energy_mean = float(np.mean(target_groups["train"]["energy"]))
    energy_scale = max(
        float(np.std(target_groups["train"]["energy"] - energy_mean)),
        1.0e-15,
    )
    force_scale = max(
        float(np.std(target_groups["train"]["force"])),
        1.0e-15,
    )
    torque_scale = max(
        float(np.std(target_groups["train"]["torque_space"])),
        1.0e-15,
    )

    dfeatures_dr = {}
    dfeatures_dtheta = {}

    for split in ("train", "validation", "test"):
        print(f"computing {split} feature derivatives ...", flush=True)
        dr, dtheta = finite_difference_feature_derivatives(
            feature_computer,
            frame_groups[split],
            feature_mask,
            x_scaler,
            position_step=args.position_step,
            rotation_step=args.rotation_step,
            quaternion_order=args.quaternion_order,
            quaternion_matrix_direction=args.quaternion_matrix_direction,
        )
        dfeatures_dr[split] = dr
        dfeatures_dtheta[split] = dtheta

    metadata = {
        "train_input": str(args.train_input),
        "validation_input": str(args.validation_input),
        "test_input": str(args.test_input),
        "n_train": len(frame_groups["train"]),
        "n_validation": len(frame_groups["validation"]),
        "n_test": len(frame_groups["test"]),
        "n_particles": n_particles,
        "raw_feature_count": int(raw_features["train"].shape[1]),
        "retained_feature_count": int(np.sum(feature_mask)),
        "normalize": True,
        "subtract_center_contribution": False,
        "max_angular": args.max_angular,
        "max_radial": args.max_radial,
        "cutoff": args.cutoff,
        "radial_width": args.radial_width,
        "basis_rcond": args.basis_rcond,
        "basis_tol": args.basis_tol,
        "variance_threshold": args.variance_threshold,
        "position_step": args.position_step,
        "rotation_step": args.rotation_step,
        "quaternion_order": args.quaternion_order,
        "quaternion_matrix_direction": (
            args.quaternion_matrix_direction
        ),
        "torque_target_frame": args.torque_target_frame,
        "energy_mean_from_train": energy_mean,
        "energy_scale_from_train": energy_scale,
        "force_scale_from_train": force_scale,
        "torque_scale_from_train": torque_scale,
    }

    arrays = {
        "metadata_json": np.array(json.dumps(metadata)),
        "feature_mask": feature_mask,
        "energy_mean": np.array(energy_mean),
        "energy_scale": np.array(energy_scale),
        "force_scale": np.array(force_scale),
        "torque_scale": np.array(torque_scale),
    }

    for split in ("train", "validation", "test"):
        arrays[f"{split}_x"] = x[split]
        arrays[f"{split}_dfeatures_dr"] = dfeatures_dr[split]
        arrays[f"{split}_dfeatures_dtheta"] = dfeatures_dtheta[split]
        arrays[f"{split}_energy"] = target_groups[split]["energy"]
        arrays[f"{split}_force"] = target_groups[split]["force"]
        arrays[f"{split}_torque_body"] = target_groups[split][
            "torque_body"
        ]
        arrays[f"{split}_torque_space"] = target_groups[split][
            "torque_space"
        ]
        arrays[f"{split}_matrices_space_to_body"] = target_groups[split][
            "matrices_space_to_body"
        ]

    np.savez(args.cache_output, **arrays)

    manifest_path = args.cache_output.with_suffix(".json")
    with manifest_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Cache written to {args.cache_output.resolve()}", flush=True)
    print(f"Manifest written to {manifest_path.resolve()}", flush=True)


def _cache_body_from_space(
    torque_space: np.ndarray,
    matrices_space_to_body: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "npij,npj->npi",
        matrices_space_to_body,
        torque_space,
    )


def _cache_predict_split(
    coefficients: np.ndarray,
    cache: dict[str, np.ndarray],
    split: str,
    *,
    energy_mean: float,
    torque_derivative_sign: float,
) -> dict[str, np.ndarray]:
    energy_centered, force, torque_space = predictions_from_coefficients(
        coefficients,
        cache[f"{split}_x"],
        cache[f"{split}_dfeatures_dr"],
        cache[f"{split}_dfeatures_dtheta"],
        torque_derivative_sign=torque_derivative_sign,
    )

    energy = energy_centered + energy_mean
    torque_body = _cache_body_from_space(
        torque_space,
        cache[f"{split}_matrices_space_to_body"],
    )

    return {
        "energy": energy,
        "force": force,
        "torque_space": torque_space,
        "torque_body": torque_body,
    }


def _cache_make_system(
    cache: dict[str, np.ndarray],
    split: str,
    *,
    energy_mean: float,
    energy_scale: float,
    force_scale: float,
    torque_scale: float,
    energy_weight: float,
    force_weight: float,
    torque_weight: float,
    torque_derivative_sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(cache[f"{split}_x"].shape[0])
    energy_centered = cache[f"{split}_energy"] - energy_mean

    return make_augmented_system(
        indices,
        cache[f"{split}_x"],
        energy_centered,
        cache[f"{split}_force"],
        cache[f"{split}_torque_space"],
        cache[f"{split}_dfeatures_dr"],
        cache[f"{split}_dfeatures_dtheta"],
        energy_scale=energy_scale,
        force_scale=force_scale,
        torque_scale=torque_scale,
        energy_weight=energy_weight,
        force_weight=force_weight,
        torque_weight=torque_weight,
        torque_derivative_sign=torque_derivative_sign,
    )


def run_cache_fit_mode(args: argparse.Namespace) -> None:
    if args.cache_input is None:
        raise ValueError("--cache-input is required for cache fit mode")

    if (
        args.energy_weight <= 0.0
        and args.force_weight <= 0.0
        and args.torque_weight <= 0.0
    ):
        raise ValueError("At least one E/F/T weight must be positive")

    args.output.mkdir(parents=True, exist_ok=True)

    data = np.load(args.cache_input, allow_pickle=False)
    cache = {
        key: data[key]
        for key in data.files
        if key != "metadata_json"
    }
    metadata = json.loads(str(data["metadata_json"].item()))

    energy_mean = float(cache["energy_mean"].item())
    energy_scale = float(cache["energy_scale"].item())
    force_scale = float(cache["force_scale"].item())
    torque_scale = float(cache["torque_scale"].item())

    train_matrix, train_target = _cache_make_system(
        cache,
        "train",
        energy_mean=energy_mean,
        energy_scale=energy_scale,
        force_scale=force_scale,
        torque_scale=torque_scale,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        torque_weight=args.torque_weight,
        torque_derivative_sign=args.torque_derivative_sign,
    )

    print("training design matrix:", train_matrix.shape, flush=True)

    if args.alpha is None:
        alpha_values = np.logspace(
            np.log10(args.alpha_min),
            np.log10(args.alpha_max),
            args.alpha_count,
        )
    else:
        alpha_values = np.asarray([args.alpha], dtype=float)

    validation_indices = np.arange(cache["validation_x"].shape[0])

    best_score = np.inf
    best_alpha = None
    scan_rows = []

    for alpha in alpha_values:
        model = Ridge(
            alpha=float(alpha),
            fit_intercept=False,
            solver="lsqr",
            tol=1.0e-10,
            max_iter=10000,
        )
        model.fit(train_matrix, train_target)

        coefficients = np.asarray(model.coef_, dtype=float)

        validation_prediction = _cache_predict_split(
            coefficients,
            cache,
            "validation",
            energy_mean=energy_mean,
            torque_derivative_sign=args.torque_derivative_sign,
        )

        score = normalized_validation_score(
            cache["validation_energy"],
            validation_prediction["energy"],
            cache["validation_force"],
            validation_prediction["force"],
            cache["validation_torque_space"],
            validation_prediction["torque_space"],
            validation_indices,
            energy_scale=energy_scale,
            force_scale=force_scale,
            torque_scale=torque_scale,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            torque_weight=args.torque_weight,
        )

        row = {
            "alpha": float(alpha),
            "validation_score": float(score),
            "validation_energy_r2": float(
                r2_score(
                    cache["validation_energy"],
                    validation_prediction["energy"],
                )
            ),
            "validation_force_r2": float(
                r2_score(
                    cache["validation_force"].reshape(-1),
                    validation_prediction["force"].reshape(-1),
                )
            ),
            "validation_torque_space_r2": float(
                r2_score(
                    cache["validation_torque_space"].reshape(-1),
                    validation_prediction["torque_space"].reshape(-1),
                )
            ),
            "validation_torque_body_r2": float(
                r2_score(
                    cache["validation_torque_body"].reshape(-1),
                    validation_prediction["torque_body"].reshape(-1),
                )
            ),
        }
        scan_rows.append(row)

        print(
            f"alpha={alpha:.3e} "
            f"score={score:.6g} "
            f"E_R2={row['validation_energy_r2']:.5f} "
            f"F_R2={row['validation_force_r2']:.5f} "
            f"T_R2={row['validation_torque_space_r2']:.5f}",
            flush=True,
        )

        if score < best_score:
            best_score = score
            best_alpha = float(alpha)

    if best_alpha is None:
        raise RuntimeError("Alpha scan produced no selected alpha")

    print(f"selected alpha: {best_alpha:.12g}", flush=True)

    # Refit on train + validation, while keeping the cache's training-only
    # feature mask, feature scaler, target mean, and target scales.
    fit_cache = {
        "fit_x": np.concatenate(
            [cache["train_x"], cache["validation_x"]],
            axis=0,
        ),
        "fit_dfeatures_dr": np.concatenate(
            [
                cache["train_dfeatures_dr"],
                cache["validation_dfeatures_dr"],
            ],
            axis=0,
        ),
        "fit_dfeatures_dtheta": np.concatenate(
            [
                cache["train_dfeatures_dtheta"],
                cache["validation_dfeatures_dtheta"],
            ],
            axis=0,
        ),
        "fit_energy": np.concatenate(
            [cache["train_energy"], cache["validation_energy"]],
            axis=0,
        ),
        "fit_force": np.concatenate(
            [cache["train_force"], cache["validation_force"]],
            axis=0,
        ),
        "fit_torque_space": np.concatenate(
            [
                cache["train_torque_space"],
                cache["validation_torque_space"],
            ],
            axis=0,
        ),
    }

    fit_indices = np.arange(fit_cache["fit_x"].shape[0])
    fit_matrix, fit_target = make_augmented_system(
        fit_indices,
        fit_cache["fit_x"],
        fit_cache["fit_energy"] - energy_mean,
        fit_cache["fit_force"],
        fit_cache["fit_torque_space"],
        fit_cache["fit_dfeatures_dr"],
        fit_cache["fit_dfeatures_dtheta"],
        energy_scale=energy_scale,
        force_scale=force_scale,
        torque_scale=torque_scale,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        torque_weight=args.torque_weight,
        torque_derivative_sign=args.torque_derivative_sign,
    )

    final_model = Ridge(
        alpha=best_alpha,
        fit_intercept=False,
        solver="lsqr",
        tol=1.0e-10,
        max_iter=10000,
    )
    final_model.fit(fit_matrix, fit_target)

    coefficients = np.asarray(final_model.coef_, dtype=float)

    test_prediction = _cache_predict_split(
        coefficients,
        cache,
        "test",
        energy_mean=energy_mean,
        torque_derivative_sign=args.torque_derivative_sign,
    )

    results = {
        "configuration": {
            "cache_input": str(args.cache_input),
            "output": str(args.output),
            "energy_weight": args.energy_weight,
            "force_weight": args.force_weight,
            "torque_weight": args.torque_weight,
            "torque_derivative_sign": args.torque_derivative_sign,
            "selected_alpha": best_alpha,
            "coefficient_count": int(coefficients.size),
            **metadata,
        },
        "test": {
            "energy": metrics(
                cache["test_energy"],
                test_prediction["energy"],
            ),
            "force_components": metrics(
                cache["test_force"],
                test_prediction["force"],
            ),
            "torque_components_space": metrics(
                cache["test_torque_space"],
                test_prediction["torque_space"],
            ),
            "torque_components_body": metrics(
                cache["test_torque_body"],
                test_prediction["torque_body"],
            ),
        },
        "alpha_scan": scan_rows,
    }

    np.savez(
        args.output / "test_predictions.npz",
        energy_target=cache["test_energy"],
        energy_prediction=test_prediction["energy"],
        force_target=cache["test_force"],
        force_prediction=test_prediction["force"],
        torque_target_space=cache["test_torque_space"],
        torque_prediction_space=test_prediction["torque_space"],
        torque_target_body=cache["test_torque_body"],
        torque_prediction_body=test_prediction["torque_body"],
        coefficients=coefficients,
        feature_mask=cache["feature_mask"],
    )

    with open(args.output / "metrics.json", "w") as handle:
        json.dump(results, handle, indent=2)

    with open(args.output / "alpha_scan.json", "w") as handle:
        json.dump(scan_rows, handle, indent=2)

    save_parity(
        cache["test_energy"],
        test_prediction["energy"],
        "Cached conservative linear fit — test energy",
        "Reference energy",
        "Predicted energy",
        args.output / "test_parity_energy.png",
    )
    save_parity(
        cache["test_force"],
        test_prediction["force"],
        "Cached conservative linear fit — test forces",
        "Reference force component",
        "Predicted force component",
        args.output / "test_parity_forces.png",
    )
    save_parity(
        cache["test_torque_body"],
        test_prediction["torque_body"],
        "Cached conservative linear fit — test body torque",
        "Reference torque component",
        "Predicted torque component",
        args.output / "test_parity_torques_body.png",
    )
    save_parity(
        cache["test_torque_space"],
        test_prediction["torque_space"],
        "Cached conservative linear fit — test space torque",
        "Reference torque component",
        "Predicted torque component",
        args.output / "test_parity_torques_space.png",
    )

    print(json.dumps(results["test"], indent=2), flush=True)
    print(f"Artifacts written to {args.output.resolve()}", flush=True)



def main() -> None:
    args = parse_args()

    if args.cache_output is not None and args.cache_input is not None:
        raise ValueError("Use only one of --cache-output or --cache-input")

    if args.cache_output is not None:
        run_cache_build_mode(args)
        return

    if args.cache_input is not None:
        run_cache_fit_mode(args)
        return

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
    if args.feature_batch_size <= 0:
        raise ValueError("--feature-batch-size must be positive")

    args.output.mkdir(parents=True, exist_ok=True)

    frames = read(args.input, ":")
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    if not frames:
        raise RuntimeError(f"No frames found in {args.input}")

    n_particles = len(frames[0])
    if any(len(frame) != n_particles for frame in frames):
        raise RuntimeError("All frames must have the same particle count")

    energy_target = np.asarray(
        [frame.get_potential_energy() for frame in frames],
        dtype=float,
    )
    force_target = np.asarray(
        [frame.get_forces() for frame in frames],
        dtype=float,
    )

    if any("torques" not in frame.arrays for frame in frames):
        raise RuntimeError("Every frame must contain a 'torques' array")

    torque_stored = np.asarray(
        [frame.arrays["torques"] for frame in frames],
        dtype=float,
    )

    if args.torque_target_frame == "body":
        torque_target_body = torque_stored
        torque_target_space = body_to_space_vectors(
            torque_target_body,
            frames,
            quaternion_order=args.quaternion_order,
            quaternion_matrix_direction=args.quaternion_matrix_direction,
        )
    else:
        torque_target_space = torque_stored
        torque_target_body = space_to_body_vectors(
            torque_target_space,
            frames,
            quaternion_order=args.quaternion_order,
            quaternion_matrix_direction=args.quaternion_matrix_direction,
        )

    print('here')

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

    feature_computer = FeatureComputer(
        calculator,
        args.feature_batch_size,
    )

    print(f"frames: {len(frames)}")
    print("computing normalized AniSOAP features ...")
    raw_features = feature_computer.raw(frames)

    finite_mask = np.isfinite(raw_features).all(axis=0)
    variance_mask = (
        np.var(raw_features, axis=0)
        >= args.variance_threshold
    )
    feature_mask = finite_mask & variance_mask
    selected_features = raw_features[:, feature_mask]

    print("raw feature shape:", raw_features.shape)
    print("retained feature shape:", selected_features.shape)

    if selected_features.shape[1] == 0:
        raise RuntimeError("No usable features remain")

    x_scaler = StandardFlexibleScaler(
        column_wise=False
    ).fit(selected_features)
    x = x_scaler.transform(selected_features)

    # Center energy explicitly so that fit_intercept=False remains valid.
    energy_mean = float(np.mean(energy_target))
    energy_centered = energy_target - energy_mean

    energy_scale = max(float(np.std(energy_centered)), 1.0e-15)
    force_scale = max(float(np.std(force_target)), 1.0e-15)
    torque_scale = max(float(np.std(torque_target_space)), 1.0e-15)

    dfeatures_dr, dfeatures_dtheta = finite_difference_feature_derivatives(
        feature_computer,
        frames,
        feature_mask,
        x_scaler,
        position_step=args.position_step,
        rotation_step=args.rotation_step,
        quaternion_order=args.quaternion_order,
        quaternion_matrix_direction=args.quaternion_matrix_direction,
    )

    frame_indices = np.arange(len(frames))
    train_indices, validation_indices = train_test_split(
        frame_indices,
        test_size=args.validation_fraction,
        random_state=args.split_seed,
    )

    train_matrix, train_target = make_augmented_system(
        train_indices,
        x,
        energy_centered,
        force_target,
        torque_target_space,
        dfeatures_dr,
        dfeatures_dtheta,
        energy_scale=energy_scale,
        force_scale=force_scale,
        torque_scale=torque_scale,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        torque_weight=args.torque_weight,
        torque_derivative_sign=args.torque_derivative_sign,
    )

    print("training design matrix:", train_matrix.shape)

    if args.alpha is None:
        alpha_values = np.logspace(
            np.log10(args.alpha_min),
            np.log10(args.alpha_max),
            args.alpha_count,
        )
    else:
        alpha_values = np.asarray([args.alpha], dtype=float)

    scan_rows = []
    best_score = np.inf
    best_alpha = None
    best_coefficients = None

    for alpha in alpha_values:
        model = Ridge(
            alpha=float(alpha),
            fit_intercept=False,
            solver="lsqr",
            tol=1.0e-10,
            max_iter=10000,
        )
        model.fit(train_matrix, train_target)

        coefficients = np.asarray(model.coef_, dtype=float)

        energy_centered_prediction, force_prediction, torque_prediction_space = (
            predictions_from_coefficients(
                coefficients,
                x,
                dfeatures_dr,
                dfeatures_dtheta,
                torque_derivative_sign=args.torque_derivative_sign,
            )
        )

        energy_prediction = energy_centered_prediction + energy_mean

        score = normalized_validation_score(
            energy_target,
            energy_prediction,
            force_target,
            force_prediction,
            torque_target_space,
            torque_prediction_space,
            validation_indices,
            energy_scale=energy_scale,
            force_scale=force_scale,
            torque_scale=torque_scale,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            torque_weight=args.torque_weight,
        )

        row = {
            "alpha": float(alpha),
            "validation_score": float(score),
            "validation_energy_r2": float(
                r2_score(
                    energy_target[validation_indices],
                    energy_prediction[validation_indices],
                )
            ),
            "validation_force_r2": float(
                r2_score(
                    force_target[validation_indices].reshape(-1),
                    force_prediction[validation_indices].reshape(-1),
                )
            ),
            "validation_torque_r2": float(
                r2_score(
                    torque_target_space[validation_indices].reshape(-1),
                    torque_prediction_space[validation_indices].reshape(-1),
                )
            ),
        }
        scan_rows.append(row)

        print(
            f"alpha={alpha:.3e} "
            f"score={score:.6g} "
            f"E_R2={row['validation_energy_r2']:.5f} "
            f"F_R2={row['validation_force_r2']:.5f} "
            f"T_R2={row['validation_torque_r2']:.5f}"
        )

        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
            best_coefficients = coefficients.copy()

    if best_alpha is None or best_coefficients is None:
        raise RuntimeError("Alpha scan produced no result")

    print(f"selected alpha: {best_alpha:.12g}")

    # Refit the chosen model using every configuration.
    full_matrix, full_target = make_augmented_system(
        frame_indices,
        x,
        energy_centered,
        force_target,
        torque_target_space,
        dfeatures_dr,
        dfeatures_dtheta,
        energy_scale=energy_scale,
        force_scale=force_scale,
        torque_scale=torque_scale,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        torque_weight=args.torque_weight,
        torque_derivative_sign=args.torque_derivative_sign,
    )

    final_model = Ridge(
        alpha=best_alpha,
        fit_intercept=False,
        solver="lsqr",
        tol=1.0e-10,
        max_iter=10000,
    )
    final_model.fit(full_matrix, full_target)
    coefficients = np.asarray(final_model.coef_, dtype=float)

    energy_centered_prediction, force_prediction, torque_prediction_space = (
        predictions_from_coefficients(
            coefficients,
            x,
            dfeatures_dr,
            dfeatures_dtheta,
            torque_derivative_sign=args.torque_derivative_sign,
        )
    )
    energy_prediction = energy_centered_prediction + energy_mean

    torque_prediction_body = space_to_body_vectors(
        torque_prediction_space,
        frames,
        quaternion_order=args.quaternion_order,
        quaternion_matrix_direction=args.quaternion_matrix_direction,
    )

    results = {
        "configuration": {
            "input": str(args.input),
            "n_frames": len(frames),
            "n_particles": n_particles,
            "raw_feature_count": int(raw_features.shape[1]),
            "retained_feature_count": int(selected_features.shape[1]),
            "normalize": True,
            "subtract_center_contribution": False,
            "max_angular": args.max_angular,
            "max_radial": args.max_radial,
            "cutoff": args.cutoff,
            "radial_width": args.radial_width,
            "position_step": args.position_step,
            "rotation_step": args.rotation_step,
            "energy_weight": args.energy_weight,
            "force_weight": args.force_weight,
            "torque_weight": args.torque_weight,
            "torque_derivative_sign": args.torque_derivative_sign,
            "selected_alpha": best_alpha,
            "validation_fraction": args.validation_fraction,
            "split_seed": args.split_seed,
            "energy_mean": energy_mean,
            "energy_scale": energy_scale,
            "force_scale": force_scale,
            "torque_scale": torque_scale,
        },
        "energy": metrics(energy_target, energy_prediction),
        "force_components": metrics(force_target, force_prediction),
        "torque_components_space": metrics(
            torque_target_space,
            torque_prediction_space,
        ),
        "torque_components_body": metrics(
            torque_target_body,
            torque_prediction_body,
        ),
        "alpha_scan": scan_rows,
    }

    np.savez(
        args.output / "predictions.npz",
        energy_target=energy_target,
        energy_prediction=energy_prediction,
        force_target=force_target,
        force_prediction=force_prediction,
        torque_target_space=torque_target_space,
        torque_prediction_space=torque_prediction_space,
        torque_target_body=torque_target_body,
        torque_prediction_body=torque_prediction_body,
        coefficients=coefficients,
        feature_mask=feature_mask,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )

    with open(args.output / "metrics.json", "w") as handle:
        json.dump(results, handle, indent=2)

    save_parity(
        energy_target,
        energy_prediction,
        "Conservative linear E/F/T fit — energy",
        "Reference energy",
        "Predicted energy",
        args.output / "parity_energy.png",
    )
    save_parity(
        force_target,
        force_prediction,
        "Conservative linear E/F/T fit — forces",
        "Reference force component",
        "Predicted force component",
        args.output / "parity_forces.png",
    )
    save_parity(
        torque_target_body,
        torque_prediction_body,
        "Conservative linear E/F/T fit — body torque",
        "Reference torque component",
        "Predicted torque component",
        args.output / "parity_torques_body.png",
    )
    save_parity(
        torque_target_space,
        torque_prediction_space,
        "Conservative linear E/F/T fit — space torque",
        "Reference torque component",
        "Predicted torque component",
        args.output / "parity_torques_space.png",
    )

    print(json.dumps(
        {
            "selected_alpha": best_alpha,
            "energy": results["energy"],
            "force_components": results["force_components"],
            "torque_components_body": results["torque_components_body"],
            "torque_components_space": results["torque_components_space"],
        },
        indent=2,
    ))
    print(f"Artifacts written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
