#!/usr/bin/env bash
set -eu

mkdir -p publication_linear_runs/logs

CACHE="publication_linear_cache/random_n10_l10_rcut5_rg2_fd1e-4.npz"

if [ ! -f "${CACHE}" ]; then
  echo "Cache not found: ${CACHE}"
  echo "Available publication_linear_cache files:"
  ls -lh publication_linear_cache || true
  exit 1
fi

python -u fit_lr_energy_force_torque.py \
  --cache-input "${CACHE}" \
  --energy-weight 10 \
  --force-weight 1 \
  --torque-weight 0 \
  --torque-derivative-sign 1 \
  --alpha-min 1e-12 \
  --alpha-max 1e-2 \
  --alpha-count 41 \
  --output publication_linear_runs/11_conservative_e10_f1_t0 \
  | tee publication_linear_runs/logs/11_conservative_e10_f1_t0.log
