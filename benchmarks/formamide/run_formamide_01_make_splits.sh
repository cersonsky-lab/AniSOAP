#!/usr/bin/env bash
set -eu

mkdir -p publication_splits

python -u make_formamide_publication_splits.py \
  --input formamide.xyz \
  --output-dir publication_splits \
  --cg-output formamide_cg.xyz \
  --n-molecules 2 \
  --energy-key interaction_energy \
  --seed 20260721 \
  --train-frac 0.70 \
  --valid-frac 0.15 \
  --diameter-scale 1.0 \
  | tee publication_splits/formamide_split_build.log
