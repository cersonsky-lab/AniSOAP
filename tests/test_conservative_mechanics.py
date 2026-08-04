import torch

from anisoap.nn import (
    apply_space_rotations,
    conservative_pair_forces_and_torques,
    pair_forces_to_atom_forces,
    rotation_vector_to_matrix,
)


def test_zero_rotation_vector_is_identity_and_differentiable():
    vector = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    matrix = rotation_vector_to_matrix(vector)

    torch.testing.assert_close(
        matrix,
        torch.eye(3, dtype=torch.float64),
    )

    # An off-diagonal functional has a nonzero first derivative at identity.
    matrix[2, 1].backward()
    assert vector.grad is not None
    assert torch.isfinite(vector.grad).all()
    assert torch.linalg.vector_norm(vector.grad) > 0


def test_space_rotation_convention():
    base = rotation_vector_to_matrix(
        torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64)
    )
    delta = torch.tensor([[0.0, 0.0, 0.4]], dtype=torch.float64)

    expected = rotation_vector_to_matrix(delta) @ base
    actual = apply_space_rotations(base, delta)
    torch.testing.assert_close(actual, expected)


def test_conservative_gradients_match_analytic_result():
    pair_vectors = torch.tensor(
        [[0.3, -0.2, 1.1], [-0.4, 0.5, 0.7]],
        dtype=torch.float64,
    )
    base_rotations = torch.eye(
        3,
        dtype=torch.float64,
    ).repeat(2, 1, 1)

    direction = torch.tensor([0.2, -0.7, 0.4], dtype=torch.float64)

    def energy_fn(pairs, rotations):
        translational = 0.5 * pairs.square().sum()
        # At identity, d/d(delta theta) of (R @ direction)_z is
        # (direction cross e_z), so torque = -dE/d(delta theta)
        # is (e_z cross direction).
        orientational = (rotations @ direction)[..., 2].sum()
        return translational + orientational

    energy, pair_forces, torques = conservative_pair_forces_and_torques(
        energy_fn,
        pair_vectors,
        base_rotations,
    )

    assert energy.ndim == 0
    torch.testing.assert_close(pair_forces, -pair_vectors)

    expected_single_torque = torch.cross(
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        direction,
        dim=0,
    )
    torch.testing.assert_close(
        torques,
        expected_single_torque.expand_as(torques),
    )


def test_pair_force_scatter_conserves_total_force():
    pair_forces = torch.tensor(
        [[1.0, 2.0, 3.0], [-0.5, 0.4, 0.2]],
        dtype=torch.float64,
    )
    centers = torch.tensor([0, 1], dtype=torch.int64)
    neighbors = torch.tensor([1, 2], dtype=torch.int64)

    atom_forces = pair_forces_to_atom_forces(
        pair_forces,
        centers,
        neighbors,
        n_atoms=3,
    )

    torch.testing.assert_close(
        atom_forces.sum(dim=0),
        torch.zeros(3, dtype=torch.float64),
    )
    torch.testing.assert_close(atom_forces[0], -pair_forces[0])
    torch.testing.assert_close(
        atom_forces[1],
        pair_forces[0] - pair_forces[1],
    )
    torch.testing.assert_close(atom_forces[2], pair_forces[1])


def test_create_graph_supports_force_training():
    stiffness = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
    pair_vectors = torch.tensor(
        [[0.3, -0.2, 1.1]],
        dtype=torch.float64,
    )
    base_rotations = torch.eye(3, dtype=torch.float64).unsqueeze(0)

    def energy_fn(pairs, rotations):
        del rotations
        return 0.5 * stiffness * pairs.square().sum()

    _, pair_forces, _ = conservative_pair_forces_and_torques(
        energy_fn,
        pair_vectors,
        base_rotations,
        create_graph=True,
    )
    loss = pair_forces.square().sum()
    gradient = torch.autograd.grad(loss, stiffness)[0]

    assert torch.isfinite(gradient)
    assert gradient.abs() > 0



def test_energy_independent_of_rotations_returns_zero_torque():
    pair_vectors = torch.tensor(
        [[0.3, -0.2, 1.1]],
        dtype=torch.float64,
    )
    base_rotations = torch.eye(3, dtype=torch.float64).unsqueeze(0)

    def energy_fn(pairs, rotations):
        del rotations
        return 0.5 * pairs.square().sum()

    _, pair_forces, torques = conservative_pair_forces_and_torques(
        energy_fn,
        pair_vectors,
        base_rotations,
    )

    torch.testing.assert_close(pair_forces, -pair_vectors)
    torch.testing.assert_close(torques, torch.zeros_like(torques))


def test_energy_independent_of_pairs_returns_zero_pair_force():
    pair_vectors = torch.tensor(
        [[0.3, -0.2, 1.1]],
        dtype=torch.float64,
    )
    base_rotations = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    direction = torch.tensor([0.2, -0.7, 0.4], dtype=torch.float64)

    def energy_fn(pairs, rotations):
        del pairs
        return (rotations @ direction)[..., 2].sum()

    _, pair_forces, torques = conservative_pair_forces_and_torques(
        energy_fn,
        pair_vectors,
        base_rotations,
    )

    torch.testing.assert_close(
        pair_forces,
        torch.zeros_like(pair_forces),
    )
    assert torch.isfinite(torques).all()
    assert torch.linalg.vector_norm(torques) > 0
