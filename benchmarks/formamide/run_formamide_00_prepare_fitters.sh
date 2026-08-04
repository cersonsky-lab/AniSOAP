#!/usr/bin/env bash
set -eu

GB_DIR="/Users/rca/source_installs/anisoap/benchmarks/two_particle_gb/force_and_torque_benchmarks"

if [ ! -f "${GB_DIR}/fit_lr_energy_force_torque.py" ]; then
  echo "Could not find ${GB_DIR}/fit_lr_energy_force_torque.py"
  echo "Edit GB_DIR in this script to point to the Gay--Berne benchmark directory."
  exit 1
fi

if [ ! -f "${GB_DIR}/fit_lr_finite_difference_eft.py" ]; then
  echo "Could not find ${GB_DIR}/fit_lr_finite_difference_eft.py"
  echo "Edit GB_DIR in this script to point to the Gay--Berne benchmark directory."
  exit 1
fi

cp "${GB_DIR}/fit_lr_energy_force_torque.py" .
cp "${GB_DIR}/fit_lr_finite_difference_eft.py" .

python -m py_compile fit_lr_energy_force_torque.py
python -m py_compile fit_lr_finite_difference_eft.py

echo "Copied and checked formamide fitters."
