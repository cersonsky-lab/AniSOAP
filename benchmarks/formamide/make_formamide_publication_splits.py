from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert atomistic formamide clusters into molecule-level "
            "ellipsoidal AniSOAP frames and make publication splits."
        )
    )
    parser.add_argument("--input", type=Path, default=Path("formamide.xyz"))
    parser.add_argument("--output-dir", type=Path, default=Path("publication_splits"))
    parser.add_argument("--cg-output", type=Path, default=Path("formamide_cg.xyz"))
    parser.add_argument("--n-molecules", type=int, default=2)
    parser.add_argument("--energy-key", type=str, default="interaction_energy")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--valid-frac", type=float, default=0.15)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--diameter-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor applied to equivalent-ellipsoid diameters inferred "
            "from principal moments."
        ),
    )
    return parser.parse_args()


def as_array(info: dict, key: str, frame_index: int) -> np.ndarray:
    if key not in info:
        raise RuntimeError(f"Frame {frame_index} missing info['{key}']")
    return np.asarray(info[key], dtype=float)


def molecule_indices(frame: Atoms, *, n_molecules: int) -> list[np.ndarray]:
    """Return atom indices for each molecule.

    The original generator wrote molecule membership as atoms.arrays["molecule_id"]
    in trajectory files, but the extended XYZ export stores the same information
    as a "labels" field in the frame metadata.  Support both representations.
    If neither is present, fall back to contiguous equal-sized molecule blocks,
    which matches the formamide generator atom ordering.
    """
    ids = None

    if "molecule_id" in frame.arrays:
        ids = np.asarray(frame.arrays["molecule_id"], dtype=int)

    elif "labels" in frame.arrays:
        ids = np.asarray(frame.arrays["labels"], dtype=float).astype(int)

    elif "labels" in frame.info:
        raw = frame.info["labels"]
        if isinstance(raw, str):
            ids = np.asarray([float(x) for x in raw.split()], dtype=float).astype(int)
        else:
            ids = np.asarray(raw, dtype=float).astype(int)

    if ids is not None:
        if ids.shape[0] != len(frame):
            raise RuntimeError(
                f"Molecule labels have length {ids.shape[0]}; expected {len(frame)}"
            )

        unique = np.unique(ids)
        if unique.size != n_molecules:
            raise RuntimeError(
                f"Found molecule labels {unique.tolist()}, but expected "
                f"{n_molecules} molecules"
            )

        return [np.flatnonzero(ids == label) for label in unique]

    if len(frame) % n_molecules != 0:
        raise RuntimeError(
            "Input frame has no molecule_id/labels metadata and cannot be "
            "split into equal contiguous molecule blocks"
        )

    atoms_per_molecule = len(frame) // n_molecules
    return [
        np.arange(i * atoms_per_molecule, (i + 1) * atoms_per_molecule)
        for i in range(n_molecules)
    ]


def equivalent_ellipsoid_diameters(
    principal_moments: np.ndarray,
    masses: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    """Infer full ellipsoid diameters from principal moments.

    For a homogeneous ellipsoid with semiaxes a,b,c and total mass M,

        I_a = M/5 (b^2 + c^2)
        I_b = M/5 (a^2 + c^2)
        I_c = M/5 (a^2 + b^2)

    so the inferred semiaxis lengths are obtained by solving this system.
    The returned values are full diameters, consistent with the c_diameter
    arrays used by the AniSOAP benchmark scripts.
    """
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise RuntimeError("Encountered non-positive molecular mass")

    ix, iy, iz = np.asarray(principal_moments, dtype=float)

    a2 = 5.0 * (iy + iz - ix) / (2.0 * total_mass)
    b2 = 5.0 * (ix + iz - iy) / (2.0 * total_mass)
    c2 = 5.0 * (ix + iy - iz) / (2.0 * total_mass)

    semiaxes = np.sqrt(np.maximum([a2, b2, c2], 1.0e-12))
    return scale * 2.0 * semiaxes


def right_handed(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=float).copy()
    if np.linalg.det(result) < 0.0:
        result[:, 2] *= -1.0
    return result


def matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrix).as_quat()
    x, y, z, w = xyzw
    return np.asarray([w, x, y, z], dtype=float)


def atomistic_to_cg(
    frame: Atoms,
    frame_index: int,
    *,
    n_molecules: int,
    energy_key: str,
    diameter_scale: float,
) -> Atoms | None:
    if int(frame.info.get("n_molecules", -1)) != n_molecules:
        return None

    indices = molecule_indices(frame, n_molecules=n_molecules)
    if len(indices) != n_molecules:
        return None

    com = as_array(frame.info, "molecular_com", frame_index)
    axes = as_array(frame.info, "principal_axes", frame_index)
    moments = as_array(frame.info, "principal_moments", frame_index)
    molecular_force = as_array(frame.info, "molecular_force", frame_index)
    molecular_torque = as_array(frame.info, "molecular_torque", frame_index)

    if com.shape != (n_molecules, 3):
        raise RuntimeError(f"Frame {frame_index} molecular_com shape {com.shape}")
    if axes.shape != (n_molecules, 3, 3):
        raise RuntimeError(f"Frame {frame_index} principal_axes shape {axes.shape}")
    if moments.shape != (n_molecules, 3):
        raise RuntimeError(f"Frame {frame_index} principal_moments shape {moments.shape}")
    if molecular_force.shape != (n_molecules, 3):
        raise RuntimeError(f"Frame {frame_index} molecular_force shape {molecular_force.shape}")
    if molecular_torque.shape != (n_molecules, 3):
        raise RuntimeError(f"Frame {frame_index} molecular_torque shape {molecular_torque.shape}")

    quaternions = []
    diameters = []
    atom_masses = frame.get_masses()

    for mol_index, atom_idx in enumerate(indices):
        # The generator stores principal axes as columns in the laboratory frame.
        # This is a body-to-space rotation matrix.
        rotation_bs = right_handed(axes[mol_index])
        quaternions.append(matrix_to_wxyz(rotation_bs))

        diameters.append(
            equivalent_ellipsoid_diameters(
                moments[mol_index],
                atom_masses[atom_idx],
                scale=diameter_scale,
            )
        )

    quaternions = np.asarray(quaternions, dtype=float)
    diameters = np.asarray(diameters, dtype=float)

    if energy_key in frame.info:
        energy = float(frame.info[energy_key])
    else:
        energy = float(frame.get_potential_energy())

    cg = Atoms(
        symbols=["X"] * n_molecules,
        positions=com,
        pbc=False,
    )
    cg.arrays["quaternions"] = quaternions
    cg.arrays["torques"] = molecular_torque
    cg.arrays["c_diameter[1]"] = diameters[:, 0]
    cg.arrays["c_diameter[2]"] = diameters[:, 1]
    cg.arrays["c_diameter[3]"] = diameters[:, 2]
    cg.arrays["molecule_index"] = np.arange(n_molecules, dtype=int)

    cg.info["source_frame"] = frame_index
    cg.info["energy_target"] = energy
    cg.info["energy_units"] = "eV"
    cg.info["force_units"] = "eV/Angstrom"
    cg.info["torque_units"] = "eV"
    cg.info["quaternion_order"] = "wxyz"
    cg.info["quaternion_matrix_direction"] = "body_to_space"
    cg.info["torque_target_frame"] = "space"
    cg.info["diameter_definition"] = "equivalent ellipsoid full diameters from principal moments"

    cg.calc = SinglePointCalculator(
        cg,
        energy=energy,
        forces=molecular_force,
    )

    return cg


def write_split(path: Path, frames: list[Atoms]) -> None:
    """Write extended XYZ with AniSOAP-compatible diameter array names."""
    path.parent.mkdir(parents=True, exist_ok=True)

    write(
        path,
        frames,
        format="extxyz",
        columns=[
            "symbols",
            "positions",
            "forces",
            "quaternions",
            "torques",
            "c_diameter[1]",
            "c_diameter[2]",
            "c_diameter[3]",
        ],
    )

    # ASE may escape bracketed property names in the header. AniSOAP expects
    # the unescaped Gay--Berne-style names.
    content = path.read_text()
    content = content.replace("c_diameter\\[1\\]", "c_diameter[1]")
    content = content.replace("c_diameter\\[2\\]", "c_diameter[2]")
    content = content.replace("c_diameter\\[3\\]", "c_diameter[3]")
    path.write_text(content)


def main() -> None:
    args = parse_args()

    if args.train_frac <= 0.0 or args.valid_frac <= 0.0:
        raise ValueError("Split fractions must be positive")
    if args.train_frac + args.valid_frac >= 1.0:
        raise ValueError("--train-frac + --valid-frac must be less than 1")
    if args.diameter_scale <= 0.0:
        raise ValueError("--diameter-scale must be positive")

    raw_frames = read(args.input, ":")
    if args.max_frames is not None:
        raw_frames = raw_frames[: args.max_frames]

    cg_frames: list[Atoms] = []
    skipped = 0

    for frame_index, frame in enumerate(raw_frames):
        cg = atomistic_to_cg(
            frame,
            frame_index,
            n_molecules=args.n_molecules,
            energy_key=args.energy_key,
            diameter_scale=args.diameter_scale,
        )
        if cg is None:
            skipped += 1
            continue
        cg_frames.append(cg)

    if not cg_frames:
        raise RuntimeError("No CG frames were produced")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_split(args.cg_output, cg_frames)

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(cg_frames))
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(round(args.train_frac * n_total))
    n_valid = int(round(args.valid_frac * n_total))
    n_test = n_total - n_train - n_valid

    train_idx = np.sort(indices[:n_train])
    valid_idx = np.sort(indices[n_train : n_train + n_valid])
    test_idx = np.sort(indices[n_train + n_valid :])

    splits = {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
    }

    for name, idx in splits.items():
        write_split(
            args.output_dir / f"formamide_{name}.xyz",
            [cg_frames[int(i)] for i in idx],
        )
        np.savetxt(
            args.output_dir / f"formamide_{name}_indices.txt",
            idx,
            fmt="%d",
        )

    energies = np.asarray([f.get_potential_energy() for f in cg_frames])
    forces = np.asarray([f.get_forces() for f in cg_frames])
    torques = np.asarray([f.arrays["torques"] for f in cg_frames])

    summary = {
        "input": str(args.input),
        "cg_output": str(args.cg_output),
        "n_raw_frames": len(raw_frames),
        "n_skipped": skipped,
        "n_cg_frames": len(cg_frames),
        "n_molecules": args.n_molecules,
        "seed": args.seed,
        "train": int(n_train),
        "valid": int(n_valid),
        "test": int(n_test),
        "energy_key": args.energy_key,
        "energy_units": "eV",
        "force_units": "eV/Angstrom",
        "torque_units": "eV",
        "quaternion_order": "wxyz",
        "quaternion_matrix_direction": "body_to_space",
        "torque_target_frame": "space",
        "diameter_scale": args.diameter_scale,
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
        "force_component_std": float(np.std(forces)),
        "torque_component_std": float(np.std(torques)),
    }

    with open(args.output_dir / "publication_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    with open(args.output_dir / "publication_manifest.csv", "w") as handle:
        handle.write("split,index,source_frame\n")
        for split, idx in splits.items():
            for i in idx:
                handle.write(
                    f"{split},{int(i)},{cg_frames[int(i)].info['source_frame']}\n"
                )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
