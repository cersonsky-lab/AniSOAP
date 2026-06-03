import pytest
import torch

from metatensor.torch import TensorMap

from anisoap.representations.ellipsoidal_density_projection import (
    EllipsoidalDensityProjection,
)


def _single_atom_graph(dtype=torch.float64):
    """Small self-pair graph matching the AniSOAP tensor-graph API."""
    R_ij = torch.zeros((1, 3), dtype=dtype, requires_grad=True)

    centers = torch.tensor([0], dtype=torch.long)
    neighbors = torch.tensor([0], dtype=torch.long)
    species = torch.tensor([0], dtype=torch.long)

    structures = torch.tensor([0], dtype=torch.long)
    atom_indices = torch.tensor([0], dtype=torch.long)

    rotations = torch.eye(3, dtype=dtype).reshape(1, 3, 3).clone()
    rotations.requires_grad_(True)

    ellipsoid_lengths = torch.tensor(
        [[0.5, 0.5, 1.0]],
        dtype=dtype,
        requires_grad=True,
    )

    return {
        "R_ij": R_ij,
        "centers": centers,
        "neighbors": neighbors,
        "species": species,
        "structures": structures,
        "atom_indices": atom_indices,
        "rotations": rotations,
        "ellipsoid_lengths": ellipsoid_lengths,
    }


def _two_atom_graph(dtype=torch.float64):
    """Two atoms with directed pair edges for testing BPNN-style per-atom features."""
    R_ij = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # self 0
            [0.0, 0.0, 0.0],  # self 1
            [1.0, 0.0, 0.0],  # 0 -> 1
            [-1.0, 0.0, 0.0], # 1 -> 0
        ],
        dtype=dtype,
        requires_grad=True,
    )

    centers = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    neighbors = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    species = torch.tensor([0, 0], dtype=torch.long)

    structures = torch.tensor([0, 0], dtype=torch.long)
    atom_indices = torch.tensor([0, 1], dtype=torch.long)

    rotations = torch.eye(3, dtype=dtype).repeat(2, 1, 1).clone()
    rotations.requires_grad_(True)

    ellipsoid_lengths = torch.tensor(
        [
            [0.5, 0.6, 0.9],
            [0.7, 0.5, 1.0],
        ],
        dtype=dtype,
        requires_grad=True,
    )

    return {
        "R_ij": R_ij,
        "centers": centers,
        "neighbors": neighbors,
        "species": species,
        "structures": structures,
        "atom_indices": atom_indices,
        "rotations": rotations,
        "ellipsoid_lengths": ellipsoid_lengths,
    }


def _calculator(max_angular=2, max_radial=2):
    return EllipsoidalDensityProjection(
        max_angular=max_angular,
        max_radial=max_radial,
        radial_basis_name="gto",
        cutoff_radius=3.0,
        radial_gaussian_width=1.5,
        basis_rcond=1e-12,
        basis_tol=1e-6,
        rotation_type="matrix",
        rotation_key="matrix",
        species=[0],
    )


def test_transform_keeps_torch_gradient_pipeline():
    calc = _calculator(max_angular=1, max_radial=1)
    graph = _single_atom_graph()

    coeffs = calc.transform(**graph, normalize=True)

    assert hasattr(coeffs, "keys")
    assert hasattr(coeffs, "blocks")
    assert hasattr(coeffs, "block")
    assert len(coeffs.blocks()) > 0

    loss = None
    for block in coeffs.blocks():
        assert torch.is_tensor(block.values)
        assert block.values.requires_grad
        block_loss = block.values.square().sum()
        loss = block_loss if loss is None else loss + block_loss

    assert loss is not None
    loss.backward()

    assert graph["R_ij"].grad is not None
    assert graph["ellipsoid_lengths"].grad is not None
    assert graph["rotations"].grad is not None

    assert torch.isfinite(graph["R_ij"].grad).all()
    assert torch.isfinite(graph["ellipsoid_lengths"].grad).all()
    assert torch.isfinite(graph["rotations"].grad).all()

    assert graph["ellipsoid_lengths"].grad.abs().sum() > 0


def test_power_spectrum_keeps_torch_gradient_pipeline():
    calc = _calculator(max_angular=2, max_radial=1)
    graph = _single_atom_graph()

    nu2 = calc.power_spectrum(
        mean_over_samples=False,
        normalize=True,
        **graph,
    )

    assert hasattr(nu2, "keys")
    assert hasattr(nu2, "blocks")
    assert hasattr(nu2, "block")
    assert len(nu2.blocks()) > 0

    loss = None
    for block in nu2.blocks():
        assert torch.is_tensor(block.values)
        assert block.values.requires_grad
        block_loss = block.values.square().sum()
        loss = block_loss if loss is None else loss + block_loss

    assert loss is not None
    loss.backward()

    assert graph["R_ij"].grad is not None
    assert graph["ellipsoid_lengths"].grad is not None
    assert graph["rotations"].grad is not None

    assert torch.isfinite(graph["R_ij"].grad).all()
    assert torch.isfinite(graph["ellipsoid_lengths"].grad).all()
    assert torch.isfinite(graph["rotations"].grad).all()


def test_anisoap_bpnn_feature_tensormap_layout_and_gradients():
    calc = _calculator(max_angular=2, max_radial=1)
    graph = _two_atom_graph()

    features = calc.power_spectrum_feature_tensor_map(**graph)

    assert hasattr(features, "keys")
    assert hasattr(features, "blocks")
    assert hasattr(features, "block")
    assert len(features.blocks()) == 1

    assert list(features.keys.names) == ["_"]
    assert features.keys.values.shape == (1, 1)

    block = features.block(0)

    assert list(block.samples.names) == ["system", "atom"]
    assert block.samples.values.shape == (2, 2)
    assert torch.equal(
        block.samples.values,
        torch.tensor([[0, 0], [0, 1]], dtype=torch.int32),
    )

    assert block.components == []
    assert list(block.properties.names) == ["property"]

    assert block.values.ndim == 2
    assert block.values.shape[0] == 2
    assert block.values.shape[1] > 0
    assert calc.shape == block.values.shape[1]

    assert torch.is_tensor(block.values)
    assert block.values.requires_grad
    assert torch.isfinite(block.values).all()

    # Simulate the first linear operation in AniSOAP-BPNN.
    weights = torch.randn(
        block.values.shape[1],
        1,
        dtype=block.values.dtype,
        device=block.values.device,
    )
    energy_like = (block.values @ weights).sum()
    energy_like.backward()

    assert graph["R_ij"].grad is not None
    assert graph["ellipsoid_lengths"].grad is not None
    assert graph["rotations"].grad is not None

    assert torch.isfinite(graph["R_ij"].grad).all()
    assert torch.isfinite(graph["ellipsoid_lengths"].grad).all()
    assert torch.isfinite(graph["rotations"].grad).all()

    assert graph["ellipsoid_lengths"].grad.abs().sum() > 0


def test_dense_power_spectrum_features_are_torch_tensors():
    calc = _calculator(max_angular=2, max_radial=1)
    graph = _two_atom_graph()

    dense, samples = calc.power_spectrum_features(
        aggregate_by_system=False,
        **graph,
    )

    assert torch.is_tensor(dense)
    assert dense.requires_grad
    assert dense.ndim == 2
    assert dense.shape[0] == 2
    assert dense.shape[1] > 0

    assert list(samples.names) in (
        ["type", "center"],
        ["system", "atom"],
    )

    loss = dense.sum()
    loss.backward()

    assert graph["R_ij"].grad is not None
    assert graph["ellipsoid_lengths"].grad is not None
    assert torch.isfinite(graph["R_ij"].grad).all()
    assert torch.isfinite(graph["ellipsoid_lengths"].grad).all()


@pytest.mark.parametrize("requires_grad_field", ["R_ij", "rotations", "ellipsoid_lengths"])
def test_no_detach_for_core_differentiable_inputs(requires_grad_field):
    calc = _calculator(max_angular=1, max_radial=1)
    graph = _single_atom_graph()

    for name in ["R_ij", "rotations", "ellipsoid_lengths"]:
        graph[name].requires_grad_(name == requires_grad_field)

    features = calc.power_spectrum_feature_tensor_map(**graph)
    loss = features.block(0).values.square().sum()
    loss.backward()

    grad = graph[requires_grad_field].grad
    assert grad is not None
    assert torch.isfinite(grad).all()
