import numpy as np
import torch
from ase import Atoms
from scipy.spatial.transform import Rotation

from anisoap.nn import LateInvariantEnergyModel
from anisoap.representations import EllipsoidalDensityProjection


def _frame():
    frame = Atoms(
        numbers=[1, 1],
        positions=[
            [0.10, -0.20, 0.00],
            [0.45, 0.30, 2.40],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )

    rotations = Rotation.from_rotvec(
        np.array(
            [
                [0.20, -0.10, 0.30],
                [-0.30, 0.40, 0.10],
            ]
        )
    )
    frame.arrays["quaternion"] = rotations.as_quat(
        canonical=True,
        scalar_first=True,
    )

    # Deliberately triaxial ellipsoids so orientation information is present.
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


def _globally_rotate(frame, rotation):
    rotated = frame.copy()
    rotated.positions[:] = rotation.apply(frame.positions)

    local = Rotation.from_quat(
        frame.arrays["quaternion"],
        scalar_first=True,
    )
    rotated.arrays["quaternion"][:] = (rotation * local).as_quat(
        canonical=True,
        scalar_first=True,
    )
    return rotated


def _rotate_one_particle(frame, particle, rotation):
    rotated = frame.copy()
    local = Rotation.from_quat(
        frame.arrays["quaternion"][particle],
        scalar_first=True,
    )
    rotated.arrays["quaternion"][particle] = (rotation * local).as_quat(
        canonical=True,
        scalar_first=True,
    )
    return rotated


def _coefficients_from_graph(calculator, graph, *, R_ij=None, rotations=None):
    return calculator.transform(
        R_ij=graph.R_ij if R_ij is None else R_ij,
        centers=graph.centers,
        neighbors=graph.neighbors,
        species=graph.species,
        structures=graph.structures,
        atom_indices=graph.atom_indices,
        rotations=graph.rotations if rotations is None else rotations,
        ellipsoid_lengths=graph.ellipsoid_lengths,
    )


def test_energy_is_invariant_under_global_rotation():
    frame = _frame()
    calculator = _calculator()
    coefficients = calculator.transform(frames=[frame])
    model = _model(coefficients)

    rotation = Rotation.from_rotvec([0.45, -0.25, 0.30])
    rotated_coefficients = calculator.transform(
        frames=[_globally_rotate(frame, rotation)]
    )

    reference = model(coefficients)
    rotated = model(rotated_coefficients)

    torch.testing.assert_close(
        rotated,
        reference,
        rtol=2.0e-6,
        atol=2.0e-8,
    )


def test_energy_changes_under_relative_particle_rotation():
    frame = _frame()
    calculator = _calculator()
    coefficients = calculator.transform(frames=[frame])
    model = _model(coefficients)

    perturbed = _rotate_one_particle(
        frame,
        particle=0,
        rotation=Rotation.from_rotvec([0.0, 0.55, 0.0]),
    )
    perturbed_coefficients = calculator.transform(frames=[perturbed])

    reference = model(coefficients)
    changed = model(perturbed_coefficients)

    assert torch.max(torch.abs(changed - reference)) > 1.0e-10


def test_center_energies_sum_to_system_energy_for_real_coefficients():
    calculator = _calculator()
    coefficients = calculator.transform(frames=[_frame()])
    model = _model(coefficients)

    center_energies, samples = model.center_energies(coefficients)
    system_energy = model(coefficients)

    assert list(samples.names) == ["type", "center"]
    assert center_energies.shape == (2,)
    assert system_energy.shape == (1,)
    torch.testing.assert_close(system_energy[0], center_energies.sum())


def test_pair_vector_gradient_is_finite_and_nonzero():
    calculator = _calculator()
    graph = calculator._graph_from_inputs(frames=[_frame()])
    pair_vectors = graph.R_ij.detach().clone().requires_grad_(True)

    coefficients = _coefficients_from_graph(
        calculator,
        graph,
        R_ij=pair_vectors,
    )
    model = _model(coefficients)
    energy = model(coefficients).sum()

    gradient = torch.autograd.grad(energy, pair_vectors)[0]

    assert gradient.shape == pair_vectors.shape
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 0


def test_rotation_matrix_gradient_is_finite_and_nonzero():
    calculator = _calculator()
    graph = calculator._graph_from_inputs(frames=[_frame()])
    rotations = graph.rotations.detach().clone().requires_grad_(True)

    coefficients = _coefficients_from_graph(
        calculator,
        graph,
        rotations=rotations,
    )
    model = _model(coefficients)
    energy = model(coefficients).sum()

    gradient = torch.autograd.grad(energy, rotations)[0]

    assert gradient.shape == rotations.shape
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 0
