import numpy as np
import torch
from ase import Atoms
from scipy.spatial.transform import Rotation

from anisoap.nn import (
    LateInvariantEnergyModel,
    conservative_pair_forces_and_torques,
    pair_forces_to_atom_forces,
)
from anisoap.representations import EllipsoidalDensityProjection


def _frame():
    frame = Atoms(
        numbers=[1, 1],
        positions=[[0.1, -0.2, 0.0], [0.45, 0.3, 2.4]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    frame.arrays["quaternion"] = Rotation.from_rotvec(
        [[0.2, -0.1, 0.3], [-0.3, 0.4, 0.1]]
    ).as_quat(canonical=True, scalar_first=True)
    frame.arrays["c_diameter[1]"] = np.array([1.0, 1.2])
    frame.arrays["c_diameter[2]"] = np.array([1.6, 1.4])
    frame.arrays["c_diameter[3]"] = np.array([3.0, 2.8])
    return frame


def _calculator():
    return EllipsoidalDensityProjection(
        max_angular=2,
        max_radial=2,
        radial_basis_name="gto",
        cutoff_radius=4.5,
        radial_gaussian_width=1.0,
        subtract_center_contribution=True,
        rotation_key="quaternion",
        rotation_type="quaternion",
        dtype=torch.float64,
    )


def _model(example):
    torch.manual_seed(7)
    return LateInvariantEnergyModel(
        example,
        active_l=(0, 2),
        hidden_channels={0: 3, 2: 3},
        max_angular=2,
    )


def _coefficients(calculator, graph, pair_vectors, rotations):
    return calculator.transform(
        R_ij=pair_vectors,
        centers=graph.centers,
        neighbors=graph.neighbors,
        species=graph.species,
        structures=graph.structures,
        atom_indices=graph.atom_indices,
        rotations=rotations,
        ellipsoid_lengths=graph.ellipsoid_lengths,
    )


def test_real_model_returns_finite_conservative_forces_and_torques():
    calculator = _calculator()
    graph = calculator._graph_from_inputs(frames=[_frame()])
    example = _coefficients(
        calculator,
        graph,
        graph.R_ij,
        graph.rotations,
    )
    model = _model(example)

    def energy_fn(pair_vectors, rotations):
        return model(_coefficients(calculator, graph, pair_vectors, rotations))

    energy, pair_forces, torques = conservative_pair_forces_and_torques(
        energy_fn,
        graph.R_ij,
        graph.rotations,
    )

    assert energy.ndim == 0
    assert pair_forces.shape == graph.R_ij.shape
    assert torques.shape == graph.rotations.shape[:-2] + (3,)
    assert torch.isfinite(pair_forces).all()
    assert torch.isfinite(torques).all()
    assert torch.linalg.vector_norm(pair_forces) > 0
    assert torch.linalg.vector_norm(torques) > 0


def test_real_pair_forces_scatter_to_zero_net_atomic_force():
    calculator = _calculator()
    graph = calculator._graph_from_inputs(frames=[_frame()])
    example = _coefficients(
        calculator,
        graph,
        graph.R_ij,
        graph.rotations,
    )
    model = _model(example)

    def energy_fn(pair_vectors, rotations):
        return model(_coefficients(calculator, graph, pair_vectors, rotations))

    _, pair_forces, _ = conservative_pair_forces_and_torques(
        energy_fn,
        graph.R_ij,
        graph.rotations,
    )
    atom_forces = pair_forces_to_atom_forces(
        pair_forces,
        graph.centers,
        graph.neighbors,
        n_atoms=len(graph.rotations),
    )

    torch.testing.assert_close(
        atom_forces.sum(dim=0),
        torch.zeros(3, dtype=atom_forces.dtype),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
