#!/usr/bin/env bash
set -eu

mkdir -p isotropic_linear_runs/logs

CACHE="isotropic_linear_cache/gb_iso_n10_l10_rcut5_rg2_fd1e-4.npz"

if [ ! -f "${CACHE}" ]; then
  echo "Cache not found: ${CACHE}"
  echo "Available isotropic_linear_cache files:"
  ls -lh isotropic_linear_cache || true
  exit 1
fi

python -u fit_lr_energy_force_torque.py \
  --cache-input "${CACHE}" \
  --energy-weight 0 \
  --force-weight 0 \
  --torque-weight 1 \
  --torque-derivative-sign 1 \
  --alpha-min 1e-12 \
  --alpha-max 1e-2 \
  --alpha-count 41 \
  --output isotropic_linear_runs/06_torque_only_diagnostic \
  | tee isotropic_linear_runs/logs/06_torque_only_diagnostic.log
