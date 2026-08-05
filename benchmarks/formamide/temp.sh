python - <<'PY'
from ase.io import read
import numpy as np

def n_mol(frame):
    return int(float(frame.info["n_molecules"]))

def molecular_force(frame):
    if "molecular_force" in frame.info:
        return np.asarray(frame.info["molecular_force"], dtype=float)
    return np.asarray(frame.get_forces(), dtype=float)

def molecular_torque(frame):
    if "molecular_torque" in frame.info:
        return np.asarray(frame.info["molecular_torque"], dtype=float)
    return np.asarray(frame.arrays["torques"], dtype=float)

for path in [
    "split_replicates/split_00/splits/formamide_train.xyz",
]:
    try:
        frames = read(path, ":")
    except Exception as exc:
        print()
        print(path)
        print("cannot read:", exc)
        continue

    if not frames:
        continue

    e_internal = np.array(
        [float(f.info["interaction_energy"]) / n_mol(f) for f in frames],
        dtype=float,
    )
    f_report = np.array([molecular_force(f) for f in frames], dtype=float)
    t_report = np.array([molecular_torque(f) for f in frames], dtype=float)
    nm = np.array([n_mol(f) for f in frames], dtype=float)

    f_internal = f_report / nm.reshape(-1, 1, 1)
    t_internal = t_report / nm.reshape(-1, 1, 1)

    print()
    print(path)
    print("n frames:", len(frames))
    print("n_molecules unique:", sorted(set(nm.astype(int).tolist())))
    print("energy internal = interaction_energy/n_molecules min/max/std:",
          e_internal.min(), e_internal.max(), e_internal.std())
    print("force report min/max/std:", f_report.min(), f_report.max(), f_report.std())
    print("force internal min/max/std:", f_internal.min(), f_internal.max(), f_internal.std())
    print("torque report min/max/std:", t_report.min(), t_report.max(), t_report.std())
    print("torque internal min/max/std:", t_internal.min(), t_internal.max(), t_internal.std())
PY
