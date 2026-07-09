from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

from gap_eft_diagnostic import (
    ENERGY_TARGET,
    build_dataset,
    build_dataset_info,
    energy_only_dataset,
    gap_dataset_with_explicit_eft_targets,
    gap_hypers,
    joined_values,
    load_dataset,
    subset_dataset,
    train_test_indices,
)
from metatrain.anisoap_gap import AnisoapGAP as GAP
from metatrain.anisoap_gap.model import (
    _apply_feature_standardization,
    _fit_feature_standardization,
    _system_feature_matrix_from_gap_features,
)
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

torch.set_default_dtype(torch.float64)


def target_stats(y):
    y = np.asarray(y).reshape(-1)
    return {
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "std": float(y.std(ddof=0)),
    }


def rmse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def rel_rmse(y_true, y_pred):
    denom = float(np.sqrt(np.mean((y_true.reshape(-1) - y_true.mean()) ** 2)))
    return rmse(y_true, y_pred) / max(denom, 1e-12)


def build_linear_features(
    *,
    xyz: Path,
    stride: int,
    test_every: int | None,
    lmax: int,
    nmax: int,
    num_sparse_points: int,
    max_frames: int | None,
):
    frames, systems = load_dataset(xyz, stride=stride)
    if max_frames is not None:
        frames = frames[:max_frames]
        systems = systems[:max_frames]

    bpnn = build_dataset(frames, systems, add_forces=True)
    gap_dataset = gap_dataset_with_explicit_eft_targets(bpnn)

    train_indices, test_indices = train_test_indices(len(gap_dataset), test_every)
    train_dataset = energy_only_dataset(subset_dataset(gap_dataset, train_indices))
    test_dataset = energy_only_dataset(subset_dataset(gap_dataset, test_indices)) if test_indices else None

    model = GAP(
        gap_hypers(
            num_sparse_points=min(num_sparse_points, len(train_dataset)),
            degree=1,
            lmax=lmax,
            nmax=nmax,
            regularizer=1e-8,
        ),
        build_dataset_info(add_forces=False),
    )

    requested = get_requested_neighbor_lists(model)

    train_systems = [
        get_system_with_neighbor_lists(sample["system"], requested)
        for sample in train_dataset
    ]

    train_features = model._soap_torch_calculator.compute(train_systems)
    train_features = train_features.keys_to_samples("center_type")
    train_features = train_features.keys_to_properties(
        ["neighbor_1_type", "neighbor_2_type"]
    )

    feature_mean, feature_std = _fit_feature_standardization(train_features)
    train_features = _apply_feature_standardization(
        train_features,
        feature_mean,
        feature_std,
    )

    X_train_raw = _system_feature_matrix_from_gap_features(train_features)
    y_train = joined_values(train_dataset, ENERGY_TARGET).reshape(-1, 1)

    # Match the trainer's second, system-feature-level standardization.
    X_mean = X_train_raw.mean(dim=0, keepdim=True)
    X_std = X_train_raw.std(dim=0, keepdim=True)
    X_std = torch.where(X_std < 1e-12, torch.ones_like(X_std), X_std)
    X_train = (X_train_raw - X_mean) / X_std

    X_test = None
    y_test = None

    if test_dataset is not None:
        test_systems = [
            get_system_with_neighbor_lists(sample["system"], requested)
            for sample in test_dataset
        ]

        test_features = model._soap_torch_calculator.compute(test_systems)
        test_features = test_features.keys_to_samples("center_type")
        test_features = test_features.keys_to_properties(
            ["neighbor_1_type", "neighbor_2_type"]
        )
        test_features = _apply_feature_standardization(
            test_features,
            feature_mean,
            feature_std,
        )

        X_test_raw = _system_feature_matrix_from_gap_features(test_features)
        X_test = (X_test_raw - X_mean) / X_std
        y_test = joined_values(test_dataset, ENERGY_TARGET).reshape(-1, 1)

    return {
        "X_train": X_train.detach().cpu().numpy(),
        "y_train": y_train.detach().cpu().numpy().reshape(-1),
        "X_test": None if X_test is None else X_test.detach().cpu().numpy(),
        "y_test": None if y_test is None else y_test.detach().cpu().numpy().reshape(-1),
        "train_indices": train_indices,
        "test_indices": test_indices,
    }


def torch_lstsq(X_train, y_train, X_test=None):
    A_train = torch.as_tensor(
        np.column_stack([np.ones(X_train.shape[0]), X_train]),
        dtype=torch.float64,
    )
    y = torch.as_tensor(y_train.reshape(-1, 1), dtype=torch.float64)

    coef = torch.linalg.lstsq(A_train, y).solution
    pred_train = (A_train @ coef).detach().cpu().numpy().reshape(-1)

    pred_test = None
    if X_test is not None:
        A_test = torch.as_tensor(
            np.column_stack([np.ones(X_test.shape[0]), X_test]),
            dtype=torch.float64,
        )
        pred_test = (A_test @ coef).detach().cpu().numpy().reshape(-1)

    return pred_train, pred_test, float(torch.linalg.norm(coef))


def torch_ridge(X_train, y_train, alpha, X_test=None):
    A_train = torch.as_tensor(
        np.column_stack([np.ones(X_train.shape[0]), X_train]),
        dtype=torch.float64,
    )
    y = torch.as_tensor(y_train.reshape(-1, 1), dtype=torch.float64)

    reg = alpha * torch.eye(A_train.shape[1], dtype=torch.float64)
    reg[0, 0] = 0.0

    lhs = A_train.T @ A_train + reg
    rhs = A_train.T @ y

    try:
        coef = torch.linalg.solve(lhs, rhs)
        solver = "solve"
    except torch._C._LinAlgError:
        coef = torch.linalg.lstsq(lhs, rhs).solution
        solver = "lstsq_fallback"

    pred_train = (A_train @ coef).detach().cpu().numpy().reshape(-1)

    pred_test = None
    if X_test is not None:
        A_test = torch.as_tensor(
            np.column_stack([np.ones(X_test.shape[0]), X_test]),
            dtype=torch.float64,
        )
        pred_test = (A_test @ coef).detach().cpu().numpy().reshape(-1)

    return pred_train, pred_test, float(torch.linalg.norm(coef)), solver


def safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
        .replace("=", "")
    )


def parity_plot(y_true, y_pred, *, title: str, output_path: Path):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    score = rmse(y_true, y_pred)
    rel = rel_rmse(y_true, y_pred)

    plt.figure()
    plt.scatter(y_true, y_pred, s=18)

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
    plt.title(f"{title}\nRMSE={score:.3e}, rel RMSE={rel:.3e}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_parity_pair(
    *,
    output_dir: Path | None,
    name: str,
    y_train,
    pred_train,
    y_test=None,
    pred_test=None,
):
    if output_dir is None:
        return

    base = output_dir / safe_name(name)
    parity_plot(
        y_train,
        pred_train,
        title=f"{name}: train",
        output_path=base / "train_energy_parity.png",
    )

    if y_test is not None and pred_test is not None:
        parity_plot(
            y_test,
            pred_test,
            title=f"{name}: test",
            output_path=base / "test_energy_parity.png",
        )


def print_scores(name, y_train, pred_train, y_test=None, pred_test=None, extra=""):
    msg = (
        f"{name:>18s} train_rmse={rmse(y_train, pred_train):.8e} "
        f"train_rel={rel_rmse(y_train, pred_train):.8e}"
    )
    if y_test is not None and pred_test is not None:
        msg += (
            f" test_rmse={rmse(y_test, pred_test):.8e} "
            f"test_rel={rel_rmse(y_test, pred_test):.8e}"
        )
    if extra:
        msg += " " + extra
    print(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", default="random_rotations.xyz")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--test-every", type=int, default=5)
    parser.add_argument("--lmax", type=int, default=5)
    parser.add_argument("--nmax", type=int, default=5)
    parser.add_argument("--num-sparse-points", type=int, default=32)
    parser.add_argument(
        "--alphas",
        default="1e-14,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for train/test parity plots.",
    )
    args = parser.parse_args()

    output_dir = None if args.output_dir is None else Path(args.output_dir).expanduser().resolve()

    data = build_linear_features(
        xyz=Path(args.xyz).expanduser().resolve(),
        stride=args.stride,
        test_every=args.test_every,
        lmax=args.lmax,
        nmax=args.nmax,
        num_sparse_points=args.num_sparse_points,
        max_frames=args.max_frames,
    )

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print("data")
    print("----")
    print("n_train:", X_train.shape[0])
    print("n_test:", None if X_test is None else X_test.shape[0])
    print("n_features:", X_train.shape[1])
    print("train_indices:", data["train_indices"][:20], "..." if len(data["train_indices"]) > 20 else "")
    print("test_indices:", data["test_indices"][:20], "..." if len(data["test_indices"]) > 20 else "")
    print("y_train stats:", target_stats(y_train))
    if y_test is not None:
        print("y_test stats:", target_stats(y_test))

    print()
    print("conditioning")
    print("------------")
    A = np.column_stack([np.ones(X_train.shape[0]), X_train])
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

    pred_train, pred_test, coef_norm = torch_lstsq(X_train, y_train, X_test)
    print_scores(
        "torch_lstsq",
        y_train,
        pred_train,
        y_test,
        pred_test,
        extra=f"coef_norm={coef_norm:.3e}",
    )
    write_parity_pair(
        output_dir=output_dir,
        name="torch_lstsq",
        y_train=y_train,
        pred_train=pred_train,
        y_test=y_test,
        pred_test=pred_test,
    )

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    for alpha in alphas:
        pred_train, pred_test, coef_norm, solver = torch_ridge(
            X_train,
            y_train,
            alpha,
            X_test,
        )
        name = f"torch_ridge {alpha:.0e}"
        print_scores(
            name,
            y_train,
            pred_train,
            y_test,
            pred_test,
            extra=f"coef_norm={coef_norm:.3e} solver={solver}",
        )
        write_parity_pair(
            output_dir=output_dir,
            name=name,
            y_train=y_train,
            pred_train=pred_train,
            y_test=y_test,
            pred_test=pred_test,
        )

    ridge_cv = RidgeCV(
        alphas=np.array(alphas, dtype=float),
        fit_intercept=True,
        scoring="neg_mean_squared_error",
    )
    ridge_cv.fit(X_train, y_train)

    pred_train = ridge_cv.predict(X_train)
    pred_test = None if X_test is None else ridge_cv.predict(X_test)

    coef_norm = float(
        np.sqrt(float(ridge_cv.intercept_ ** 2) + np.sum(ridge_cv.coef_ ** 2))
    )

    print_scores(
        "sklearn_RidgeCV",
        y_train,
        pred_train,
        y_test,
        pred_test,
        extra=f"alpha={ridge_cv.alpha_:.3e} coef_norm={coef_norm:.3e}",
    )
    write_parity_pair(
        output_dir=output_dir,
        name=f"sklearn_RidgeCV_alpha_{ridge_cv.alpha_:.0e}",
        y_train=y_train,
        pred_train=pred_train,
        y_test=y_test,
        pred_test=pred_test,
    )

    if output_dir is not None:
        print()
        print("wrote parity plots to", output_dir)


if __name__ == "__main__":
    main()
