#!/usr/bin/env bash
set -eu

python -u fit_lr_energy_force_torque.py \
  --cache-input publication_linear_cache/random_n10_l10_rcut5_rg2_fd1e-4.npz \
  --energy-weight 100 \
  --force-weight 1 \
  --torque-weight 0.25 \
  --torque-derivative-sign 1 \
  --alpha-min 1e-12 \
  --alpha-max 1e-2 \
  --alpha-count 41 \
  --output publication_linear_runs/10_conservative_e100_f1_t025 \
  | tee publication_linear_runs/logs/10_conservative_e100_f1_t025.log
