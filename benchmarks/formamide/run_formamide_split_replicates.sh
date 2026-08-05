#!/usr/bin/env bash
set -eu

ROOT="split_replicates"

MAX_RADIAL=10
MAX_ANGULAR=10
CUTOFF=5.0
RADIAL_WIDTH=2.5
DIAMETER_SCALE=0.7
BASIS_RCOND=1e-6
BASIS_TOL=1e-2
POSITION_STEP=1e-4
ROTATION_STEP=1e-4
FEATURE_BATCH_SIZE=128

ALPHA_MIN=1e-8
ALPHA_MAX=1e3
ALPHA_COUNT=24

FORCE_REBUILD="${FORCE_REBUILD:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

build_cache () {
  split_id="$1"
  geometry="$2"

  split_dir="${ROOT}/${split_id}"
  splits="${split_dir}/splits"

  if [ "${geometry}" = "anisotropic" ]; then
    cache_dir="${split_dir}/anisotropic_cache"
    cache="${cache_dir}/formamide_aniso_n10_l10_rcut5_rg2p5_dscale0p7.npz"
    iso_args=""
  elif [ "${geometry}" = "isotropic" ]; then
    cache_dir="${split_dir}/isotropic_cache"
    cache="${cache_dir}/formamide_iso_n10_l10_rcut5_rg2p5_dscale0p7.npz"
    iso_args="--isotropic-geometry volume_equivalent"
  else
    echo "unknown geometry: ${geometry}" >&2
    exit 1
  fi

  mkdir -p "${cache_dir}" "${split_dir}/logs"

  if [ -f "${cache}" ] && [ "${FORCE_REBUILD}" != "1" ]; then
    echo "cache exists, skipping: ${cache}"
    return
  fi

  echo
  echo "============================================================"
  echo "Building ${geometry} cache for ${split_id}"
  echo "============================================================"

  # shellcheck disable=SC2086
  python -u fit_lr_energy_force_torque.py \
    --cache-output "${cache}" \
    --train-input "${splits}/formamide_train.xyz" \
    --validation-input "${splits}/formamide_valid.xyz" \
    --test-input "${splits}/formamide_test.xyz" \
    ${iso_args} \
    --max-radial "${MAX_RADIAL}" \
    --max-angular "${MAX_ANGULAR}" \
    --cutoff "${CUTOFF}" \
    --radial-width "${RADIAL_WIDTH}" \
    --diameter-scale "${DIAMETER_SCALE}" \
    --basis-rcond "${BASIS_RCOND}" \
    --basis-tol "${BASIS_TOL}" \
    --position-step "${POSITION_STEP}" \
    --rotation-step "${ROTATION_STEP}" \
    --feature-batch-size "${FEATURE_BATCH_SIZE}" \
    --quaternion-order wxyz \
    --quaternion-matrix-direction body_to_space \
    --torque-target-frame space \
    | tee "${split_dir}/logs/build_${geometry}_cache.log"
}

run_cached_fit () {
  split_id="$1"
  geometry="$2"
  run_name="$3"
  e_weight="$4"
  f_weight="$5"
  t_weight="$6"

  split_dir="${ROOT}/${split_id}"

  if [ "${geometry}" = "anisotropic" ]; then
    cache="${split_dir}/anisotropic_cache/formamide_aniso_n10_l10_rcut5_rg2p5_dscale0p7.npz"
    out_root="${split_dir}/anisotropic_runs"
  elif [ "${geometry}" = "isotropic" ]; then
    cache="${split_dir}/isotropic_cache/formamide_iso_n10_l10_rcut5_rg2p5_dscale0p7.npz"
    out_root="${split_dir}/isotropic_runs"
  else
    echo "unknown geometry: ${geometry}" >&2
    exit 1
  fi

  out="${out_root}/${run_name}"
  mkdir -p "${out_root}" "${split_dir}/logs"

  if [ -f "${out}/metrics.json" ] && [ "${FORCE_RERUN}" != "1" ]; then
    echo "metrics exist, skipping: ${out}"
    return
  fi

  rm -rf "${out}"

  echo
  echo "============================================================"
  echo "Running ${split_id} ${geometry} ${run_name}: beta=(${e_weight},${f_weight},${t_weight})"
  echo "============================================================"

  python -u fit_lr_energy_force_torque.py \
    --cache-input "${cache}" \
    --energy-weight "${e_weight}" \
    --force-weight "${f_weight}" \
    --torque-weight "${t_weight}" \
    --torque-derivative-sign -1 \
    --alpha-min "${ALPHA_MIN}" \
    --alpha-max "${ALPHA_MAX}" \
    --alpha-count "${ALPHA_COUNT}" \
    --output "${out}" \
    | tee "${split_dir}/logs/${geometry}_${run_name}.log"
}

run_split () {
  split_id="$1"

  build_cache "${split_id}" anisotropic
  build_cache "${split_id}" isotropic

  run_cached_fit "${split_id}" anisotropic 01_energy_only_descriptor_derivative 1 0 0
  run_cached_fit "${split_id}" anisotropic 02_joint_e10_f1_t025 10 1 0.25
  run_cached_fit "${split_id}" anisotropic 03_force_trained 0 1 0
  run_cached_fit "${split_id}" anisotropic 04_torque_trained 0 0 1

  run_cached_fit "${split_id}" isotropic 01_energy_only_descriptor_derivative 1 0 0
  run_cached_fit "${split_id}" isotropic 02_joint_e30_f1_t0 30 1 0
}

run_split split_0$1
# for split_id in split_00 split_01 split_02 split_03 split_04; do
#   run_split "${split_id}"
# done
