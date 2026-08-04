#!/usr/bin/env bash
set -eu

mkdir -p isotropic_linear_cache isotropic_linear_runs/logs

python -u fit_lr_energy_force_torque.py \
  --isotropic-geometry volume_equivalent \
  --cache-output isotropic_linear_cache/formamide_iso_n10_l10_rcut8_rg1_fd1e-4.npz \
  --train-input publication_splits/formamide_train.xyz \
  --validation-input publication_splits/formamide_valid.xyz \
  --test-input publication_splits/formamide_test.xyz \
  --max-angular 10 \
  --max-radial 10 \
  --cutoff 8.0 \
  --radial-width 1.0 \
  --basis-rcond 1e-6 \
  --basis-tol 1e-2 \
  --position-step 1e-4 \
  --rotation-step 1e-4 \
  --feature-batch-size 128 \
  --quaternion-order wxyz \
  --quaternion-matrix-direction body_to_space \
  --torque-target-frame space \
  | tee isotropic_linear_cache/formamide_iso_n10_l10_rcut8_rg1_fd1e-4.log
