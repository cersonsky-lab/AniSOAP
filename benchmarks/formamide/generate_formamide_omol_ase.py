#!/usr/bin/env python3
"""
Generate formamide dimer/trimer configurations, evaluate them with Meta FAIR
Chemistry's OMol-trained UMA MLIP, and store everything in an ASE trajectory.

Each saved frame contains:
  * total potential energy and atomic forces in a SinglePointCalculator
  * atoms.arrays["molecule_id"]       : molecule membership per atom
  * atoms.info["molecular_com"]       : shape (n_mol, 3), Angstrom
  * atoms.info["inertia_tensor"]      : shape (n_mol, 3, 3), amu Angstrom^2
  * atoms.info["principal_moments"]   : shape (n_mol, 3), amu Angstrom^2
  * atoms.info["principal_axes"]      : shape (n_mol, 3, 3)
  * atoms.info["molecular_force"]     : shape (n_mol, 3), eV/Angstrom
  * atoms.info["molecular_torque"]    : shape (n_mol, 3), eV
  * atoms.info["monomer_energies"]    : isolated, equally distorted monomers, eV
  * atoms.info["interaction_energy"]  : E(cluster) - sum E(monomers), eV
  * for trimers, optionally:
      pair_energies, pair_interaction_energies, three_body_energy

Notes
-----
1. OMol/UMA requires total charge and spin multiplicity in Atoms.info.
   Neutral formamide clusters are singlets: charge=0, spin=1.
2. The trajectory is non-periodic. Molecules are centered around the origin.
3. Intramolecular vibration is represented by small random Cartesian
   displacements with rigid translation and infinitesimal rigid rotation
   projected out. This is a controlled near-equilibrium sampler, not a
   Boltzmann-exact vibrational distribution.
4. Computing monomer and pair decompositions requires extra MLIP evaluations.
   For 10,000 clusters, --decomposition full can be much more expensive than
   --decomposition none.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase.units import kB

try:
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
except ImportError as exc:
    raise SystemExit(
        "fairchem-core is required. Install it in a suitable PyTorch environment:\n"
        "  pip install fairchem-core ase numpy\n"
        "Then authenticate for the gated UMA checkpoint:\n"
        "  huggingface-cli login"
    ) from exc


# A near-planar neutral formamide reference structure, HCONH2.
# Atom order: C, O, N, H(carbonyl), H(N), H(N)
# Coordinates are in Angstrom and are subsequently centered at the COM.
FORMAMIDE_SYMBOLS = ["C", "O", "N", "H", "H", "H"]
FORMAMIDE_POSITIONS = np.array(
    [
        [ 0.000000,  0.000000,  0.000000],  # C
        [ 1.223000,  0.000000,  0.000000],  # O
        [-0.744000,  1.126000,  0.000000],  # N
        [-0.493000, -0.965000,  0.000000],  # H-C
        [-1.754000,  1.087000,  0.000000],  # H-N
        [-0.283000,  2.025000,  0.000000],  # H-N
    ],
    dtype=float,
)

# Used only to set relative amplitudes of random internal displacement.
# The generator projects out rigid translation and rotation afterward.
BOND_PAIRS = ((0, 1), (0, 2), (0, 3), (2, 4), (2, 5))


@dataclass(frozen=True)
class SamplingSettings:
    min_com_distance: float = 3.0
    max_com_distance: float = 6.5
    min_atom_distance: float = 1.2
    vibration_rms: float = 0.05
    vibration_max_atom: float = 0.10
    compact_probability: float = 0.70
    max_placement_attempts: int = 500


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation in SO(3), generated from a unit quaternion."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w),     2 * (x*z + y*w)],
            [2 * (x*y + z*w),     1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y)],
        ]
    )


def center_of_mass(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return np.average(positions, axis=0, weights=masses)


def inertia_tensor(
    positions: np.ndarray, masses: np.ndarray, com: np.ndarray | None = None
) -> np.ndarray:
    """Return the Cartesian inertia tensor in amu Angstrom^2."""
    if com is None:
        com = center_of_mass(positions, masses)
    r = positions - com
    rr = np.einsum("ni,nj->nij", r, r)
    r2 = np.einsum("ni,ni->n", r, r)
    eye = np.eye(3)
    return np.sum(masses[:, None, None] * (r2[:, None, None] * eye - rr), axis=0)


def remove_rigid_components(
    displacement: np.ndarray, positions: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    """
    Project mass-weighted rigid translation and infinitesimal rotation out of
    a Cartesian displacement.
    """
    disp = displacement.copy()
    total_mass = masses.sum()

    # Remove center-of-mass translation.
    disp -= np.sum(masses[:, None] * disp, axis=0) / total_mass

    com = center_of_mass(positions, masses)
    r = positions - com

    # Find the infinitesimal rotation omega minimizing
    # sum_i m_i |disp_i - omega x r_i|^2.
    inertia = inertia_tensor(positions, masses, com)
    angular_rhs = np.sum(np.cross(r, masses[:, None] * disp), axis=0)
    omega = np.linalg.pinv(inertia, rcond=1e-12) @ angular_rhs
    disp -= np.cross(omega[None, :], r)

    # Numerical cleanup of translation.
    disp -= np.sum(masses[:, None] * disp, axis=0) / total_mass
    return disp


def vibrate_monomer(
    reference: Atoms,
    rng: np.random.Generator,
    target_rms: float,
    max_atom_displacement: float,
) -> Atoms:
    """
    Add a small internal distortion. Heavy atoms move less than hydrogens,
    and rigid translation/rotation are projected out.
    """
    mol = reference.copy()
    masses = mol.get_masses()
    positions = mol.get_positions()

    # Approximately mass-weighted random vibration. Add a correlated component
    # along bonds so bond stretches and bends are represented.
    disp = rng.normal(size=positions.shape) / np.sqrt(masses[:, None])

    for i, j in BOND_PAIRS:
        direction = positions[j] - positions[i]
        direction /= np.linalg.norm(direction)
        amplitude = rng.normal()
        disp[i] -= 0.5 * amplitude * direction / math.sqrt(masses[i])
        disp[j] += 0.5 * amplitude * direction / math.sqrt(masses[j])

    disp = remove_rigid_components(disp, positions, masses)

    rms = math.sqrt(np.mean(np.sum(disp**2, axis=1)))
    if rms > 0:
        # Draw a half-normal amplitude around the requested RMS.
        requested = min(
            abs(rng.normal(loc=target_rms, scale=0.35 * target_rms)),
            2.0 * target_rms,
        )
        disp *= requested / rms

    largest = np.max(np.linalg.norm(disp, axis=1))
    if largest > max_atom_displacement:
        disp *= max_atom_displacement / largest

    mol.set_positions(positions + disp)

    # Preserve the original COM exactly before cluster placement.
    old_com = center_of_mass(positions, masses)
    new_com = center_of_mass(mol.get_positions(), masses)
    mol.translate(old_com - new_com)
    return mol


def build_reference_formamide() -> Atoms:
    mol = Atoms(FORMAMIDE_SYMBOLS, positions=FORMAMIDE_POSITIONS, pbc=False)
    mol.translate(-mol.get_center_of_mass())
    mol.info.update({"charge": 0, "spin": 1})
    return mol


def sample_radius(rng: np.random.Generator, settings: SamplingSettings) -> float:
    """
    Mixture distribution: mostly compact hydrogen-bonding distances, with a
    tail extending toward weakly interacting/dissociated configurations.
    """
    if rng.random() < settings.compact_probability:
        # Truncated normal concentrated around contact/H-bond distances.
        for _ in range(100):
            r = rng.normal(loc=3.35, scale=0.45)
            if settings.min_com_distance <= r <= min(4.8, settings.max_com_distance):
                return float(r)
    # Volume-uniform radial sampling for broader coverage.
    lo3 = settings.min_com_distance**3
    hi3 = settings.max_com_distance**3
    return float((lo3 + rng.random() * (hi3 - lo3)) ** (1.0 / 3.0))


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def minimum_inter_molecular_distance(
    positions_a: np.ndarray, positions_b: np.ndarray
) -> float:
    delta = positions_a[:, None, :] - positions_b[None, :, :]
    return float(np.sqrt(np.min(np.sum(delta * delta, axis=-1))))


def make_cluster(
    n_molecules: int,
    reference: Atoms,
    rng: np.random.Generator,
    settings: SamplingSettings,
) -> Atoms:
    """Generate one non-periodic dimer or trimer with rejection of hard clashes."""
    if n_molecules not in (2, 3):
        raise ValueError("Only dimers and trimers are supported.")

    for _ in range(settings.max_placement_attempts):
        monomers: list[Atoms] = []
        target_coms = [np.zeros(3)]

        if n_molecules == 2:
            target_coms.append(sample_radius(rng, settings) * random_unit_vector(rng))
        else:
            # Place molecule 2 relative to molecule 1.
            c2 = sample_radius(rng, settings) * random_unit_vector(rng)

            # Place molecule 3 relative to either molecule 1 or 2. This samples
            # triangular, chain-like, and partially dissociated trimers.
            anchor = np.zeros(3) if rng.random() < 0.5 else c2
            c3 = anchor + sample_radius(rng, settings) * random_unit_vector(rng)
            target_coms.extend([c2, c3])

        valid = True
        for mol_index, target_com in enumerate(target_coms):
            mol = vibrate_monomer(
                reference,
                rng,
                target_rms=settings.vibration_rms,
                max_atom_displacement=settings.vibration_max_atom,
            )
            rotation = random_rotation_matrix(rng)
            pos = mol.get_positions() @ rotation.T

            masses = mol.get_masses()
            pos -= center_of_mass(pos, masses)
            pos += target_com
            mol.set_positions(pos)

            for previous in monomers:
                if (
                    minimum_inter_molecular_distance(
                        mol.get_positions(), previous.get_positions()
                    )
                    < settings.min_atom_distance
                ):
                    valid = False
                    break
            if not valid:
                break
            monomers.append(mol)

        if valid:
            cluster = monomers[0].copy()
            molecule_id = np.zeros(len(monomers[0]), dtype=np.int32)
            for mol_index, mol in enumerate(monomers[1:], start=1):
                cluster += mol
                molecule_id = np.concatenate(
                    [molecule_id, np.full(len(mol), mol_index, dtype=np.int32)]
                )

            # Recenter the entire cluster for numerical convenience.
            cluster.translate(-cluster.get_center_of_mass())
            cluster.set_pbc(False)
            cluster.set_cell(np.zeros((3, 3)))
            cluster.arrays["molecule_id"] = molecule_id
            cluster.info.update(
                {
                    "charge": 0,
                    "spin": 1,
                    "n_molecules": n_molecules,
                }
            )
            return cluster

    raise RuntimeError(
        "Failed to place a clash-free cluster. Consider reducing "
        "--min-atom-distance or --min-com-distance."
    )


def molecule_indices(atoms: Atoms) -> list[np.ndarray]:
    ids = np.asarray(atoms.arrays["molecule_id"], dtype=int)
    return [np.flatnonzero(ids == i) for i in range(int(ids.max()) + 1)]


def molecular_geometry_metadata(atoms: Atoms) -> dict[str, np.ndarray]:
    positions = atoms.get_positions()
    masses = atoms.get_masses()

    coms = []
    inertias = []
    moments = []
    axes = []
    labels = np.zeros(len(atoms))
    for idx in molecule_indices(atoms):
        labels[idx] = len(coms)
        com = center_of_mass(positions[idx], masses[idx])
        tensor = inertia_tensor(positions[idx], masses[idx], com)
        eigenvalues, eigenvectors = np.linalg.eigh(tensor)
        coms.append(com)
        inertias.append(tensor)
        moments.append(eigenvalues)
        # Columns are principal axes in the laboratory Cartesian frame.
        axes.append(eigenvectors)

    return {
        "molecular_com": np.asarray(coms),
        "inertia_tensor": np.asarray(inertias),
        "principal_moments": np.asarray(moments),
        "principal_axes": np.asarray(axes),
        "labels": np.asarray(labels)
    }


def molecular_force_and_torque(
    atoms: Atoms, forces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positions = atoms.get_positions()
    masses = atoms.get_masses()
    net_forces = []
    torques = []

    for idx in molecule_indices(atoms):
        com = center_of_mass(positions[idx], masses[idx])
        rel = positions[idx] - com
        f = forces[idx]
        net_forces.append(np.sum(f, axis=0))
        torques.append(np.sum(np.cross(rel, f), axis=0))

    return np.asarray(net_forces), np.asarray(torques)


def subset_atoms(atoms: Atoms, molecule_numbers: Sequence[int]) -> Atoms:
    ids = np.asarray(atoms.arrays["molecule_id"], dtype=int)
    mask = np.isin(ids, np.asarray(molecule_numbers, dtype=int))
    sub = atoms[mask]
    sub.set_pbc(False)
    sub.set_cell(np.zeros((3, 3)))
    sub.info.update({"charge": 0, "spin": 1})
    return sub


def evaluate_energy_forces(atoms: Atoms, calculator) -> tuple[float, np.ndarray]:
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    return energy, forces


def energy_decomposition(
    atoms: Atoms,
    calculator,
    cluster_energy: float,
    mode: str,
) -> dict[str, np.ndarray | float]:
    """
    Compute isolated distorted-monomer references and optional trimer
    pair/three-body terms.

    For a trimer:
      pair_interaction_ij = E_ij - E_i - E_j
      E_3body = E_123 - sum(E_ij) + sum(E_i)
    """
    n_mol = int(atoms.info["n_molecules"])
    if mode == "none":
        return {}

    monomer_energies = np.array(
        [
            evaluate_energy_forces(subset_atoms(atoms, [i]), calculator)[0]
            for i in range(n_mol)
        ]
    )
    result: dict[str, np.ndarray | float] = {
        "monomer_energies": monomer_energies,
        "interaction_energy": float(cluster_energy - monomer_energies.sum()),
    }

    if n_mol == 3 and mode == "full":
        pairs = ((0, 1), (0, 2), (1, 2))
        pair_energies = np.array(
            [
                evaluate_energy_forces(subset_atoms(atoms, pair), calculator)[0]
                for pair in pairs
            ]
        )
        pair_interactions = np.array(
            [
                pair_energies[k] - monomer_energies[i] - monomer_energies[j]
                for k, (i, j) in enumerate(pairs)
            ]
        )
        three_body = (
            cluster_energy - pair_energies.sum() + monomer_energies.sum()
        )
        result.update(
            {
                "pair_molecule_ids": np.asarray(pairs, dtype=np.int32),
                "pair_energies": pair_energies,
                "pair_interaction_energies": pair_interactions,
                "three_body_energy": float(three_body),
            }
        )

    return result


def attach_stored_results(
    atoms: Atoms,
    energy: float,
    forces: np.ndarray,
    extra_info: dict,
) -> Atoms:
    """Detach the live MLIP and attach portable ASE single-point results."""
    stored = atoms.copy()
    stored.info.update(extra_info)
    stored.calc = SinglePointCalculator(stored, energy=energy, forces=forces)
    return stored


def verify_saved_frame(atoms: Atoms) -> None:
    """Cheap consistency checks before writing."""
    n_mol = int(atoms.info["n_molecules"])
    required_shapes = {
        "molecular_com": (n_mol, 3),
        "inertia_tensor": (n_mol, 3, 3),
        "principal_moments": (n_mol, 3),
        "principal_axes": (n_mol, 3, 3),
        "molecular_force": (n_mol, 3),
        "molecular_torque": (n_mol, 3),
    }
    for key, shape in required_shapes.items():
        arr = np.asarray(atoms.info[key])
        if arr.shape != shape or not np.all(np.isfinite(arr)):
            raise ValueError(f"{key} has invalid shape/data: {arr.shape}")

    forces = atoms.get_forces()
    if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
        raise ValueError("Invalid atomic forces.")
    if not np.isfinite(atoms.get_potential_energy()):
        raise ValueError("Invalid potential energy.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=Path("formamide_clusters.xyz"))
    parser.add_argument("--n-configs", type=int, default=10_000)
    parser.add_argument(
        "--trimer-fraction",
        type=float,
        default=0.5,
        help="Probability that a generated configuration is a trimer.",
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--model",
        default="uma-s-1p2",
        help="FAIR Chemistry pretrained model name.",
    )
    parser.add_argument(
        "--decomposition",
        choices=("none", "monomers", "full"),
        default="full",
        help=(
            "none: cluster E/F only; monomers: monomer and total interaction "
            "energies; full: also trimer pair and three-body energies."
        ),
    )
    parser.add_argument("--min-com-distance", type=float, default=3.5)
    parser.add_argument("--max-com-distance", type=float, default=6.5)
    parser.add_argument("--min-atom-distance", type=float, default=1.2)
    parser.add_argument("--vibration-rms", type=float, default=0.05)
    parser.add_argument("--vibration-max-atom", type=float, default=0.10)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing trajectory instead of replacing it.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress after this many saved configurations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_configs <= 0:
        raise SystemExit("--n-configs must be positive.")
    if not 0.0 <= args.trimer_fraction <= 1.0:
        raise SystemExit("--trimer-fraction must be between 0 and 1.")
    if args.min_com_distance >= args.max_com_distance:
        raise SystemExit("--min-com-distance must be less than --max-com-distance.")

    settings = SamplingSettings(
        min_com_distance=args.min_com_distance,
        max_com_distance=args.max_com_distance,
        min_atom_distance=args.min_atom_distance,
        vibration_rms=args.vibration_rms,
        vibration_max_atom=args.vibration_max_atom,
    )

    rng = np.random.default_rng(args.seed)
    reference = build_reference_formamide()

    # FAIR Chemistry currently uses a seed during predictor construction.
    predictor_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    predictor = pretrained_mlip.get_predict_unit(
        args.model,
        device=args.device,
        seed=predictor_seed,
    )
    calculator = FAIRChemCalculator(predictor, task_name="omol")

    mode = "a" if args.append else "w"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_dimers = 0
    n_trimers = 0
    failures = 0

    frames = []

    config_index = 0
    while config_index < args.n_configs:
        n_molecules = 3 if rng.random() < args.trimer_fraction else 2
        try:
            cluster = make_cluster(
                n_molecules=n_molecules,
                reference=reference,
                rng=rng,
                settings=settings,
            )

            energy, forces = evaluate_energy_forces(cluster, calculator)
            geometry = molecular_geometry_metadata(cluster)
            net_force, torque = molecular_force_and_torque(cluster, forces)
            decomposition = energy_decomposition(
                cluster,
                calculator,
                cluster_energy=energy,
                mode=args.decomposition,
            )

            info = {
                **geometry,
                **decomposition,
                "molecular_force": net_force,
                "molecular_torque": torque,
                "config_index": config_index,
                "generator_seed": args.seed,
                "mlip_model": args.model,
                "mlip_task": "omol",
                "energy_units": "eV",
                "force_units": "eV/Angstrom",
                "torque_units": "eV",
                "length_units": "Angstrom",
                "inertia_units": "amu*Angstrom^2",
                "vibration_rms_target": args.vibration_rms,
            }

            stored = attach_stored_results(cluster, energy, forces, info)
            verify_saved_frame(stored)
            frames.append(stored)

            config_index += 1
            n_dimers += n_molecules == 2
            n_trimers += n_molecules == 3

            if (
                args.progress_every > 0
                and config_index % args.progress_every == 0
            ):
                print(
                    f"saved={config_index}/{args.n_configs} "
                    f"dimers={n_dimers} trimers={n_trimers} "
                    f"placement/evaluation_failures={failures}",
                    flush=True,
                )

        except (RuntimeError, ValueError, FloatingPointError) as exc:
            failures += 1
            print(
                f"Skipping failed sample attempt {failures}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            # Avoid an accidental infinite loop in a badly chosen parameter regime.
            if failures > max(1000, 10 * args.n_configs):
                raise RuntimeError("Too many failed sample attempts.") from exc

    print(
        f"Finished: {args.output} contains {args.n_configs} new frames "
        f"({n_dimers} dimers, {n_trimers} trimers)."
    )

    write(args.output, frames)

if __name__ == "__main__":
    main()
