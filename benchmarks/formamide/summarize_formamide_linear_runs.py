from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("publication_linear_runs")

RUNS = [
    ("Finite differences of learned energy", "(1,0,0)", "01_energy_only_finite_difference"),
    ("Descriptor derivatives", "(1,0,0)", "02_energy_only_descriptor_derivative"),
    ("Force-trained", "(0,1,0)", "03_force_trained"),
    ("Torque-trained", "(0,0,1)", "04_torque_trained"),
    ("Joint", "(10,1,0.25)", "05_joint_e10_f1_t025"),
    ("Joint", "(30,1,0)", "06_joint_e30_f1_t0"),
    ("Joint", "(30,1,0.1)", "07_joint_e30_f1_t01"),
    ("Joint", "(100,1,0)", "08_joint_e100_f1_t0"),
    ("Joint", "(100,1,0.1)", "09_joint_e100_f1_t01"),
    ("Joint", "(100,1,0.25)", "10_joint_e100_f1_t025"),
]


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    return data["test"] if "test" in data else data


print("| model | beta | alpha | E R² | F R² | tau R² | E RMSE | F RMSE | tau RMSE |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

for model, beta, dirname in RUNS:
    path = ROOT / dirname / "metrics.json"

    if not path.exists():
        print(f"| {model} | {beta} | missing | | | | | | |")
        continue

    data = json.loads(path.read_text())
    block = data["test"] if "test" in data else data
    cfg = data.get("configuration", {})

    alpha = cfg.get("selected_alpha")
    alpha_text = f"{alpha:.3g}" if isinstance(alpha, (int, float)) else "?"

    e = block["energy"]
    f = block["force_components"]

    if "torque_components_body" in block:
        tau = block["torque_components_body"]
    else:
        tau = block["torque_components_space"]

    print(
        f"| {model} | {beta} | {alpha_text} | "
        f"{e['r2']:.4f} | {f['r2']:.4f} | {tau['r2']:.4f} | "
        f"{e['rmse']:.4g} | {f['rmse']:.4g} | {tau['rmse']:.4g} |"
    )
