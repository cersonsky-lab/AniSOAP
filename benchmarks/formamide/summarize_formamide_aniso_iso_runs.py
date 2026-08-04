from __future__ import annotations

import json
import math
from pathlib import Path


ANISO_DIR = Path("publication_linear_runs")
ISO_DIR = Path("isotropic_linear_runs")

ANISO_MODELS = [
    ("01_energy_only_finite_difference", "Finite differences of learned energy", "(1,0,0)"),
    ("02_energy_only_descriptor_derivative", "Descriptor derivatives", "(1,0,0)"),
    ("03_force_trained", "Force-trained", "(0,1,0)"),
    ("04_torque_trained", "Torque-trained", "(0,0,1)"),
    ("05_joint_e10_f1_t025", "Joint", "(10,1,0.25)"),
    ("06_joint_e30_f1_t0", "Joint", "(30,1,0)"),
    ("07_joint_e30_f1_t01", "Joint", "(30,1,0.1)"),
    ("08_joint_e100_f1_t0", "Joint", "(100,1,0)"),
    ("09_joint_e100_f1_t01", "Joint", "(100,1,0.1)"),
    ("10_joint_e100_f1_t025", "Joint", "(100,1,0.25)"),
]

ISO_MODELS = [
    ("01_energy_only_descriptor_derivative", "Descriptor derivatives", "(1,0,0)"),
    ("02_force_trained", "Force-trained", "(0,1,0)"),
    ("03_joint_e10_f1_t0", "Joint", "(10,1,0)"),
    ("04_joint_e30_f1_t0", "Joint", "(30,1,0)"),
    ("05_joint_e100_f1_t0", "Joint", "(100,1,0)"),
]


def load_metrics(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def get_test_block(metrics: dict) -> dict:
    if "test" in metrics:
        return metrics["test"]
    return metrics


def metric(test: dict, observable: str, key: str) -> float:
    return float(test[observable][key])


def selected_alpha(metrics: dict) -> float:
    cfg = metrics.get("configuration", {})
    value = cfg.get("selected_alpha", float("nan"))
    return float(value)


def config_value(metrics: dict, key: str, default="") -> str:
    value = metrics.get("configuration", {}).get(key, default)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def fmt_float(x: float, digits: int = 4) -> str:
    if not math.isfinite(x):
        return ""
    if abs(x) >= 1.0e4 or (abs(x) < 1.0e-3 and x != 0.0):
        return f"{x:.3g}"
    return f"{x:.{digits}f}"


def fmt_alpha(x: float) -> str:
    if not math.isfinite(x):
        return ""
    if x == 0.0:
        return "0"
    if abs(x) < 1.0e-3 or abs(x) >= 1.0e3:
        return f"{x:.3g}"
    return f"{x:.4g}"


def row_from_run(
    geometry: str,
    root: Path,
    run_name: str,
    label: str,
    beta: str,
) -> dict | None:
    path = root / run_name / "metrics.json"
    if not path.exists():
        print(f"missing: {path}")
        return None

    metrics = load_metrics(path)
    test = get_test_block(metrics)

    torque_key = (
        "torque_components_body"
        if "torque_components_body" in test
        else "torque_components_space"
    )

    return {
        "geometry": geometry,
        "run": run_name,
        "model": label,
        "beta": beta,
        "alpha": selected_alpha(metrics),
        "E_R2": metric(test, "energy", "r2"),
        "F_R2": metric(test, "force_components", "r2"),
        "tau_R2": metric(test, torque_key, "r2"),
        "E_RMSE": metric(test, "energy", "rmse"),
        "F_RMSE": metric(test, "force_components", "rmse"),
        "tau_RMSE": metric(test, torque_key, "rmse"),
        "cutoff": config_value(metrics, "cutoff"),
        "radial_width": config_value(metrics, "radial_width"),
        "diameter_scale": config_value(metrics, "diameter_scale", "1"),
        "torque_key": torque_key,
    }


def markdown_table(rows: list[dict]) -> str:
    headers = [
        "geometry",
        "model",
        "beta",
        "alpha",
        "E R²",
        "F R²",
        "tau R²",
        "E RMSE",
        "F RMSE",
        "tau RMSE",
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["geometry"],
                    row["model"],
                    row["beta"],
                    fmt_alpha(row["alpha"]),
                    fmt_float(row["E_R2"]),
                    fmt_float(row["F_R2"]),
                    fmt_float(row["tau_R2"]),
                    fmt_float(row["E_RMSE"]),
                    fmt_float(row["F_RMSE"]),
                    fmt_float(row["tau_RMSE"]),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def compact_comparison(rows: list[dict]) -> str:
    by_key = {(r["geometry"], r["run"]): r for r in rows}

    pairs = [
        (
            "Energy-only descriptor",
            ("anisotropic", "02_energy_only_descriptor_derivative"),
            ("isotropic", "01_energy_only_descriptor_derivative"),
        ),
        (
            "Force-trained",
            ("anisotropic", "03_force_trained"),
            ("isotropic", "02_force_trained"),
        ),
        (
            "Joint (10,1,0)",
            None,
            ("isotropic", "03_joint_e10_f1_t0"),
        ),
        (
            "Joint (30,1,0)",
            ("anisotropic", "06_joint_e30_f1_t0"),
            ("isotropic", "04_joint_e30_f1_t0"),
        ),
        (
            "Joint (100,1,0)",
            ("anisotropic", "08_joint_e100_f1_t0"),
            ("isotropic", "05_joint_e100_f1_t0"),
        ),
    ]

    lines = []
    lines.append("## Direct anisotropic/isotropic comparisons\n")
    lines.append(
        "| comparison | geometry | E R² | F R² | tau R² | E RMSE | F RMSE | tau RMSE |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for label, aniso_key, iso_key in pairs:
        if aniso_key is not None and aniso_key in by_key:
            r = by_key[aniso_key]
            lines.append(
                f"| {label} | anisotropic | "
                f"{fmt_float(r['E_R2'])} | {fmt_float(r['F_R2'])} | {fmt_float(r['tau_R2'])} | "
                f"{fmt_float(r['E_RMSE'])} | {fmt_float(r['F_RMSE'])} | {fmt_float(r['tau_RMSE'])} |"
            )

        if iso_key in by_key:
            r = by_key[iso_key]
            lines.append(
                f"| {label} | isotropic | "
                f"{fmt_float(r['E_R2'])} | {fmt_float(r['F_R2'])} | {fmt_float(r['tau_R2'])} | "
                f"{fmt_float(r['E_RMSE'])} | {fmt_float(r['F_RMSE'])} | {fmt_float(r['tau_RMSE'])} |"
            )

    return "\n".join(lines)


def main() -> None:
    rows: list[dict] = []

    for run_name, label, beta in ANISO_MODELS:
        row = row_from_run("anisotropic", ANISO_DIR, run_name, label, beta)
        if row is not None:
            rows.append(row)

    for run_name, label, beta in ISO_MODELS:
        row = row_from_run("isotropic", ISO_DIR, run_name, label, beta)
        if row is not None:
            rows.append(row)

    text = []
    text.append("# Formamide linear AniSOAP anisotropic/isotropic summary\n")

    aniso_meta = next((r for r in rows if r["geometry"] == "anisotropic"), None)
    iso_meta = next((r for r in rows if r["geometry"] == "isotropic"), None)

    if aniso_meta:
        text.append(
            "Anisotropic publication descriptor: "
            f"cutoff={aniso_meta['cutoff']}, "
            f"radial_width={aniso_meta['radial_width']}, "
            f"diameter_scale={aniso_meta['diameter_scale']}.\n"
        )

    if iso_meta:
        text.append(
            "Isotropic baseline descriptor: "
            f"cutoff={iso_meta['cutoff']}, "
            f"radial_width={iso_meta['radial_width']}, "
            f"diameter_scale={iso_meta['diameter_scale']} "
            "(volume-equivalent spherical geometry if recorded in cache metadata).\n"
        )

    text.append("## Full table\n")
    text.append(markdown_table(rows))
    text.append("")
    text.append(compact_comparison(rows))

    # Best balanced anisotropic model among joint rows.
    joint_rows = [
        r for r in rows
        if r["geometry"] == "anisotropic"
        and r["model"] == "Joint"
        and r["F_R2"] > -100
        and r["tau_R2"] > -100
    ]

    if joint_rows:
        best_joint = max(
            joint_rows,
            key=lambda r: min(r["E_R2"], r["F_R2"], r["tau_R2"]),
        )
        text.append("\n## Best balanced anisotropic joint model\n")
        text.append(
            f"`{best_joint['run']}` with beta={best_joint['beta']}: "
            f"E R²={fmt_float(best_joint['E_R2'])}, "
            f"F R²={fmt_float(best_joint['F_R2'])}, "
            f"tau R²={fmt_float(best_joint['tau_R2'])}."
        )

    out = "\n".join(text)
    out_path = Path("publication_linear_runs/formamide_aniso_iso_summary.md")
    out_path.write_text(out)

    print(out)
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
