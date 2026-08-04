#!/usr/bin/env bash
set -eu

mkdir -p isotropic_linear_cache isotropic_linear_runs/logs

python -u fit_lr_energy_force_torque.py \
  --isotropic-geometry volume_equivalent \
  --cache-output isotropic_linear_cache/gb_iso_n10_l10_rcut5_rg2_fd1e-4.npz \
  --train-input publication_splits/random_train.xyz \
  --validation-input publication_splits/random_valid.xyz \
  --test-input publication_splits/random_test.xyz \
  --max-angular 10 \
  --max-radial 10 \
  --cutoff 5.0 \
  --radial-width 2.0 \
  --basis-rcond 1e-6 \
  --basis-tol 1e-2 \
  --position-step 1e-4 \
  --rotation-step 1e-4 \
  --feature-batch-size 128 \
  --quaternion-order wxyz \
  --quaternion-matrix-direction space_to_body \
  --torque-target-frame body \
  | tee isotropic_linear_cache/gb_iso_n10_l10_rcut5_rg2_fd1e-4.log
