import torch

from anisoap.representations import EllipsoidalDensityProjection


def test_features_from_torch_state_differentiates_positions_orientations_and_lengths():
    calc = EllipsoidalDensityProjection(
        max_angular=2,
        max_radial=1,
        radial_basis_name="gto",
        cutoff_radius=3.0,
        radial_gaussian_width=1.5,
        rotation_type="matrix",
        species=[0],
    )

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, -0.2],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    centers = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    neighbors = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    species = torch.tensor([0, 0], dtype=torch.long)

    q_raw = torch.tensor(
        [
            [1.0, 0.1, 0.0, 0.0],
            [1.0, 0.0, 0.2, 0.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    ellipsoid_lengths = torch.tensor(
        [
            [0.5, 0.6, 0.9],
            [0.7, 0.5, 1.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    features = calc.features_from_torch_state(
        positions=positions,
        centers=centers,
        neighbors=neighbors,
        species=species,
        orientation_quaternions=q_raw,
        ellipsoid_lengths=ellipsoid_lengths,
    )

    block = features.block(0)

    assert list(block.samples.names) == ["system", "atom"]
    assert list(block.properties.names) == ["property"]
    assert torch.is_tensor(block.values)
    assert block.values.requires_grad
    assert torch.isfinite(block.values).all()

    loss = block.values.square().sum()

    grad_positions, grad_quaternions, grad_lengths = torch.autograd.grad(
        loss,
        [positions, q_raw, ellipsoid_lengths],
    )

    assert torch.isfinite(grad_positions).all()
    assert torch.isfinite(grad_quaternions).all()
    assert torch.isfinite(grad_lengths).all()

    assert grad_positions.abs().sum() > 0
    assert grad_quaternions.abs().sum() > 0
    assert grad_lengths.abs().sum() > 0


def test_features_from_torch_state_accepts_rotation_matrices_directly():
    calc = EllipsoidalDensityProjection(
        max_angular=1,
        max_radial=1,
        radial_basis_name="gto",
        cutoff_radius=3.0,
        radial_gaussian_width=1.5,
        rotation_type="matrix",
        species=[0],
    )

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.1],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    centers = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    neighbors = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    species = torch.tensor([0, 0], dtype=torch.long)

    rotations = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    rotations.requires_grad_(True)

    ellipsoid_lengths = torch.tensor(
        [
            [0.5, 0.6, 0.9],
            [0.7, 0.5, 1.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    features = calc.features_from_torch_state(
        positions=positions,
        centers=centers,
        neighbors=neighbors,
        species=species,
        rotations=rotations,
        ellipsoid_lengths=ellipsoid_lengths,
    )

    loss = features.block(0).values.sum()
    grad_positions, grad_rotations, grad_lengths = torch.autograd.grad(
        loss,
        [positions, rotations, ellipsoid_lengths],
    )

    assert torch.isfinite(grad_positions).all()
    assert torch.isfinite(grad_rotations).all()
    assert torch.isfinite(grad_lengths).all()
