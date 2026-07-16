#!/usr/bin/env bash
set -euo pipefail

scripts=(
  run_pub_01_energy_only_finite_difference.sh
  run_pub_02_joint_e10_f1_t025.sh
  run_pub_03_energy_only_derivative_pipeline.sh
  run_pub_04_force_only.sh
  run_pub_05_torque_only.sh
)

for script in "${scripts[@]}"; do
  echo
  echo "============================================================"
  echo "Running ${script}"
  echo "============================================================"
  "./${script}"
done
