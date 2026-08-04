#!/usr/bin/env bash
set -eu

mkdir -p publication_linear_runs/logs

CACHE="publication_linear_cache/formamide_n10_l10_rcut5_rg2p5_dscale0p7.npz"

run_one () {
  name="$1"
  e="$2"
  f="$3"
  t="$4"

  echo
  echo "============================================================"
  echo "Running ${name}: beta=(${e},${f},${t})"
  echo "============================================================"

  rm -rf "publication_linear_runs/${name}"

  python -u fit_lr_energy_force_torque.py \
    --cache-input "${CACHE}" \
    --energy-weight "${e}" \
    --force-weight "${f}" \
    --torque-weight "${t}" \
    --torque-derivative-sign -1 \
    --alpha-min 1e-8 \
    --alpha-max 1e3 \
    --alpha-count 24 \
    --output "publication_linear_runs/${name}" \
    | tee "publication_linear_runs/logs/${name}.log"
}

run_one 02_energy_only_descriptor_derivative 1 0 0
run_one 03_force_trained 0 1 0
run_one 04_torque_trained 0 0 1
run_one 05_joint_e10_f1_t025 10 1 0.25
run_one 06_joint_e30_f1_t0 30 1 0
run_one 07_joint_e30_f1_t01 30 1 0.1
run_one 08_joint_e100_f1_t0 100 1 0
run_one 09_joint_e100_f1_t01 100 1 0.1
run_one 10_joint_e100_f1_t025 100 1 0.25
