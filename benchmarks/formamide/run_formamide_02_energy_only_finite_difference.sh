#!/usr/bin/env bash
set -eu

mkdir -p publication_linear_runs/logs

python -u fit_lr_finite_difference_eft.py \
  --train-input publication_splits/formamide_train.xyz \
  --validation-input publication_splits/formamide_valid.xyz \
  --test-input publication_splits/formamide_test.xyz \
  --max-angular 10 \
  --max-radial 10 \
  --cutoff 5.0 \
  --diameter-scale 0.7 \
  --radial-width 2.5 \
  --basis-rcond 1e-6 \
  --basis-tol 1e-2 \
  --position-step 1e-4 \
  --rotation-step 1e-4 \
  --fd-batch-size 128 \
  --alpha-min 1e-12 \
  --alpha-max 1e-2 \
  --alpha-count 41 \
  --quaternion-order wxyz \
  --quaternion-matrix-direction body_to_space \
  --torque-target-frame space \
  --output publication_linear_runs/01_energy_only_finite_difference \
  | tee publication_linear_runs/logs/01_energy_only_finite_difference.log
