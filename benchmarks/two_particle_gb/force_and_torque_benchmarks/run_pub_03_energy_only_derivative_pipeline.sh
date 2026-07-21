#!/usr/bin/env bash
set -eu

python -u fit_lr_energy_force_torque.py \
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
  --energy-weight 1 \
  --force-weight 0 \
  --torque-weight 0 \
  --torque-derivative-sign 1 \
  --alpha-min 1e-12 \
  --alpha-max 1e-2 \
  --alpha-count 41 \
  --quaternion-order wxyz \
  --quaternion-matrix-direction space_to_body \
  --torque-target-frame body \
  --output publication_linear_runs/03_energy_only_derivative_pipeline \ \
  | tee publication_linear_runs/logs/03_energy_only_derivative_pipeline.log
