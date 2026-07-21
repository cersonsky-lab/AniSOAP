from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba


ROOT = Path("publication_linear_runs")
FIGDIR = Path("paper_figures/linear_anisoap_gb")
PARITY_DIR = FIGDIR / "si_parity"
FIGDIR.mkdir(parents=True, exist_ok=True)
PARITY_DIR.mkdir(parents=True, exist_ok=True)

# Internal directory mapping only.
MODEL_DIRS = {
    "fd_energy_only": "01_energy_only_finite_difference",
    "b_1_0_0": "03_energy_only_derivative_pipeline",
    "b_0_1_0": "04_force_only",
    "b_0_0_1": "05_torque_only",
    "b_10_1_025": "02_joint_e10_f1_t025",
    "b_30_1_0": "06_conservative_e30_f1_t0",
    "b_30_1_01": "08_conservative_e30_f1_t01",
    "b_100_1_0": "07_conservative_e100_f1_t0",
    "b_100_1_01": "09_conservative_e100_f1_t01",
    "b_100_1_025": "10_conservative_e100_f1_t025",
}

BETA_LABELS = {
    "fd_energy_only": r"$(\beta_E,\beta_F,\beta_\tau)=(1,0,0)$",
    "b_1_0_0": r"$(\beta_E,\beta_F,\beta_\tau)=(1,0,0)$",
    "b_0_1_0": r"$(\beta_E,\beta_F,\beta_\tau)=(0,1,0)$",
    "b_0_0_1": r"$(\beta_E,\beta_F,\beta_\tau)=(0,0,1)$",
    "b_10_1_025": r"$(\beta_E,\beta_F,\beta_\tau)=(10,1,0.25)$",
    "b_30_1_0": r"$(\beta_E,\beta_F,\beta_\tau)=(30,1,0)$",
    "b_30_1_01": r"$(\beta_E,\beta_F,\beta_\tau)=(30,1,0.1)$",
    "b_100_1_0": r"$(\beta_E,\beta_F,\beta_\tau)=(100,1,0)$",
    "b_100_1_01": r"$(\beta_E,\beta_F,\beta_\tau)=(100,1,0.1)$",
    "b_100_1_025": r"$(\beta_E,\beta_F,\beta_\tau)=(100,1,0.25)$",
}

# Descriptor-derivative models only for the learning-weight tradeoff plot.
TRADEOFF_MODELS = [
    "b_1_0_0",
    "b_10_1_025",
    "b_30_1_0",
    "b_30_1_01",
    "b_100_1_0",
    "b_100_1_01",
    "b_100_1_025",
]

# Supplemental-information parity plots, one row per model.
SI_PARITY_MODELS = [
    "fd_energy_only",
    "b_1_0_0",
    "b_0_1_0",
    "b_0_0_1",
    "b_10_1_025",
    "b_30_1_0",
    "b_30_1_01",
    "b_100_1_0",
    "b_100_1_01",
    "b_100_1_025",
]

ENERGY_COLOR = "#4C78A8"
FORCE_COLOR = "#54A24B"
TORQUE_COLOR = "#F58518"

ENERGY_BG = to_rgba(ENERGY_COLOR, 0.10)
FORCE_BG = to_rgba(FORCE_COLOR, 0.10)
TORQUE_BG = to_rgba(TORQUE_COLOR, 0.10)


def load_metrics(model_key: str) -> dict:
    path = ROOT / MODEL_DIRS[model_key] / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text())
    return data["test"] if "test" in data else data


def load_predictions(model_key: str) -> dict[str, np.ndarray]:
    path = ROOT / MODEL_DIRS[model_key] / "test_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    return dict(np.load(path))


def metric_row(model_key: str) -> dict[str, float]:
    block = load_metrics(model_key)
    return {
        "E_R2": float(block["energy"]["r2"]),
        "F_R2": float(block["force_components"]["r2"]),
        "T_R2": float(block["torque_components_body"]["r2"]),
        "E_RMSE": float(block["energy"]["rmse"]),
        "F_RMSE": float(block["force_components"]["rmse"]),
        "T_RMSE": float(block["torque_components_body"]["rmse"]),
    }


def style_axis(ax) -> None:
    ax.grid(alpha=0.25)
    ax.tick_params(direction="out")


def save_learning_weight_tradeoff() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    markers = ["*", "s", "^", "D", "v", "P", "X"]
    colors = ['b','orange','g','r','purple','brown','pink','cyan','grey']
    handles = []
    labels = []

    filled_handles = []
    filled_labels = []

    for marker, c, model_key in zip(markers, colors, TRADEOFF_MODELS):
        row = metric_row(model_key)

        for r2, filled in zip([row["F_R2"], row["T_R2"]], [True, False]):

            handle = ax.scatter(
                row["E_R2"],
                r2,
                s=70,
                marker=marker,
                alpha=0.95,
                label=BETA_LABELS[model_key],
                facecolor=c if filled else 'none',
                edgecolor=c,
            )
            if filled:
                handles.append(handle)
                labels.append(BETA_LABELS[model_key])

    filled_handles.append(ax.scatter(0,0,s=70, marker='o', label='Force', facecolor='grey',edgecolor='grey'))
    filled_handles.append(ax.scatter(0,0,s=70, marker='o', label='Torue', facecolor='none',edgecolor='grey'))
    ax.set_xlabel(r"Energy $R^2$")
    ax.set_ylabel(r"Force/torque $R^2$")
    ax.set_xlim(0.92, 1.00)
    ax.set_ylim(0.73, 0.84)
    style_axis(ax)

    l1 = ax.legend(
        handles,
        labels,
        frameon=False,
        loc="lower left",
        fontsize=8,
        ncol=1,
        handletextpad=0.6,
        borderaxespad=0.4,
    )

    l2 = ax.legend(filled_handles, 
                   ['Force','Torque'], 
        frameon=False,
        loc="upper right",
        fontsize=8,
        ncol=1,
        handletextpad=0.6,
        borderaxespad=0.4,)
    ax.add_artist(l1)

    fig.tight_layout()
    fig.savefig(FIGDIR / "linear_anisoap_learning_weight_tradeoff.pdf")
    fig.savefig(FIGDIR / "linear_anisoap_learning_weight_tradeoff.png", dpi=300)
    plt.close(fig)


def parity_limits(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    target = np.asarray(target, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    finite = np.isfinite(target) & np.isfinite(prediction)

    target = target[finite]
    prediction = prediction[finite]

    lo = min(float(target.min()), float(prediction.min()))
    hi = max(float(target.max()), float(prediction.max()))

    span = hi - lo
    pad = 0.04 * span if span > 0 else 0.04
    return lo - pad, hi + pad


def parity_panel(
    ax,
    target: np.ndarray,
    prediction: np.ndarray,
    observable_name: str,
    facecolor,
    point_color: str,
) -> None:
    target = np.asarray(target, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)

    finite = np.isfinite(target) & np.isfinite(prediction)
    target = target[finite]
    prediction = prediction[finite]

    lo, hi = parity_limits(target, prediction)

    ax.set_facecolor(facecolor)
    ax.scatter(
        target,
        prediction,
        s=8,
        alpha=0.45,
        color=point_color,
        edgecolors="none",
    )
    ax.plot([lo, hi], [lo, hi], linewidth=1.0, color="black", alpha=0.85)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(observable_name)
    ax.ticklabel_format(style="scientific", axis="both", scilimits=(-3, 3))
    style_axis(ax)


def save_si_parity_triptych(model_key: str) -> None:
    pred = load_predictions(model_key)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(9.2, 3.0),
        sharex=False,
        sharey=False,
    )

    parity_panel(
        axes[0],
        pred["energy_target"],
        pred["energy_prediction"],
        r"Energy ($\epsilon$)",
        ENERGY_BG,
        ENERGY_COLOR,
    )
    parity_panel(
        axes[1],
        pred["force_target"],
        pred["force_prediction"],
        r"Force components ($\epsilon/\sigma_0$)",
        FORCE_BG,
        FORCE_COLOR,
    )
    parity_panel(
        axes[2],
        pred["torque_target_body"],
        pred["torque_prediction_body"],
        r"Torque components ($\epsilon$)",
        TORQUE_BG,
        TORQUE_COLOR,
    )

    fig.supxlabel("Reference")
    fig.supylabel("Predicted")
    fig.suptitle(BETA_LABELS[model_key], y=1.03)

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    stem = f"linear_anisoap_parity_{model_key}"
    fig.savefig(PARITY_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PARITY_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    save_learning_weight_tradeoff()

    for model_key in SI_PARITY_MODELS:
        save_si_parity_triptych(model_key)

    print(f"Wrote tradeoff plot to {FIGDIR.resolve()}")
    print(f"Wrote SI parity plots to {PARITY_DIR.resolve()}")


if __name__ == "__main__":
    main()
