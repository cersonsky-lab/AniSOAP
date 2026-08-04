from __future__ import annotations

import json
from pathlib import Path


RUNS = [
    # Existing anisotropic production runs
    ("anisotropic", "energy-only", "(1,0,0)", Path("publication_linear_runs/03_energy_only_derivative_pipeline/metrics.json")),
    ("anisotropic", "force-only", "(0,1,0)", Path("publication_linear_runs/04_force_only/metrics.json")),
    ("anisotropic", "torque-only", "(0,0,1)", Path("publication_linear_runs/05_torque_only/metrics.json")),
    ("anisotropic", "joint", "(10,1,0.25)", Path("publication_linear_runs/02_joint_e10_f1_t025/metrics.json")),
    ("anisotropic", "joint", "(10,1,0)", Path("publication_linear_runs/11_conservative_e10_f1_t0/metrics.json")),
    ("anisotropic", "joint", "(30,1,0)", Path("publication_linear_runs/06_conservative_e30_f1_t0/metrics.json")),
    ("anisotropic", "joint", "(100,1,0)", Path("publication_linear_runs/07_conservative_e100_f1_t0/metrics.json")),

    # Isotropic matched runs
    ("isotropic", "energy-only", "(1,0,0)", Path("isotropic_linear_runs/01_energy_only_descriptor_derivative/metrics.json")),
    ("isotropic", "force-only", "(0,1,0)", Path("isotropic_linear_runs/02_force_trained/metrics.json")),
    ("isotropic", "torque-only diagnostic", "(0,0,1)", Path("isotropic_linear_runs/06_torque_only_diagnostic/metrics.json")),
    ("isotropic", "joint", "(10,1,0)", Path("isotropic_linear_runs/03_joint_e10_f1_t0/metrics.json")),
    ("isotropic", "joint", "(30,1,0)", Path("isotropic_linear_runs/04_joint_e30_f1_t0/metrics.json")),
    ("isotropic", "joint", "(100,1,0)", Path("isotropic_linear_runs/05_joint_e100_f1_t0/metrics.json")),
]


def get_block(data: dict) -> dict:
    return data["test"] if "test" in data else data


def get_alpha(data: dict) -> str:
    cfg = data.get("configuration", {})
    alpha = cfg.get("selected_alpha", data.get("selected_alpha"))
    if isinstance(alpha, (float, int)):
        return f"{alpha:.3g}"
    return "?"


def get_torque(block: dict) -> dict:
    if "torque_components_body" in block:
        return block["torque_components_body"]
    if "torque_components_space" in block:
        return block["torque_components_space"]
    raise KeyError("No torque_components_body or torque_components_space block found")


print("| geometry | model | beta | alpha | E R² | F R² | torque R² | E RMSE | F RMSE | torque RMSE |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

for geometry, model, beta, path in RUNS:
    if not path.exists():
        print(f"| {geometry} | {model} | {beta} | missing | | | | | | |")
        continue

    data = json.loads(path.read_text())
    block = get_block(data)

    e = block["energy"]
    f = block["force_components"]
    t = get_torque(block)

    print(
        f"| {geometry} | {model} | {beta} | {get_alpha(data)} | "
        f"{e['r2']:.4f} | {f['r2']:.4f} | {t['r2']:.4f} | "
        f"{e['rmse']:.4g} | {f['rmse']:.4g} | {t['rmse']:.4g} |"
    )
