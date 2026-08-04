#!/usr/bin/env bash
set -eu

./run_formamide_00_prepare_fitters.sh
./run_formamide_01_make_splits.sh
./run_formamide_02_energy_only_finite_difference.sh
./run_formamide_03_build_linear_cache.sh
./run_formamide_04_weight_sweep_from_cache.sh

python summarize_formamide_linear_runs.py | tee publication_linear_runs/formamide_linear_summary.md
