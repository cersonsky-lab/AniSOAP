#!/usr/bin/env bash
set -eu

./run_formamide_iso_01_build_linear_cache.sh
./run_formamide_iso_02_weight_sweep_from_cache.sh
