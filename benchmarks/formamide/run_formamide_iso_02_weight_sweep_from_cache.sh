#!/usr/bin/env bash
set -eu

mkdir -p isotropic_linear_runs/logs

CACHE="isotropic_linear_cache/formamide_iso_n10_l10_rcut8_rg1_fd1e-4.npz"

run_fit () {
  name="$1"
  e="$2"
  f="$3"
  t="$4"

  echo
  echo "============================================================"
  echo "Running isotropic ${name}: beta=(${e},${f},${t})"
  echo "============================================================"

  python -u fit_lr_energy_force_torque.py \
    --cache-input "${CACHE}" \
    --energy-weight "${e}" \
    --force-weight "${f}" \
    --torque-weight "${t}" \
    --torque-derivative-sign 1 \
    --alpha-min 1e-12 \
    --alpha-max 1e-2 \
    --alpha-count 41 \
    --output "isotropic_linear_runs/${name}" \
    | tee "isotropic_linear_runs/logs/${name}.log"
}

run_fit "01_energy_only_descriptor_derivative" 1 0 0
run_fit "02_force_trained" 0 1 0
run_fit "03_joint_e10_f1_t0" 10 1 0
run_fit "04_joint_e30_f1_t0" 30 1 0
run_fit "05_joint_e100_f1_t0" 100 1 0
