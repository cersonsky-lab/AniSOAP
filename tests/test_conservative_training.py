import torch

from anisoap.nn import EnergyForceTorqueLoss, conservative_training_step


def test_weighted_energy_force_torque_loss():
    loss_fn = EnergyForceTorqueLoss(
        energy_weight=2.0,
        force_weight=3.0,
        torque_weight=4.0,
    )

    predicted_energy = torch.tensor(2.0)
    target_energy = torch.tensor(1.0)

    predicted_forces = torch.tensor([[1.0, 0.0, -1.0]])
    target_forces = torch.zeros_like(predicted_forces)

    predicted_torques = torch.tensor([[0.0, 2.0, 0.0]])
    target_torques = torch.zeros_like(predicted_torques)

    total, terms = loss_fn(
        predicted_energy=predicted_energy,
        target_energy=target_energy,
        predicted_forces=predicted_forces,
        target_forces=target_forces,
        predicted_torques=predicted_torques,
        target_torques=target_torques,
    )

    expected_energy = torch.tensor(1.0)
    expected_force = torch.tensor(2.0 / 3.0)
    expected_torque = torch.tensor(4.0 / 3.0)
    expected_total = (
        2.0 * expected_energy
        + 3.0 * expected_force
        + 4.0 * expected_torque
    )

    torch.testing.assert_close(terms["energy"], expected_energy)
    torch.testing.assert_close(terms["force"], expected_force)
    torch.testing.assert_close(terms["torque"], expected_torque)
    torch.testing.assert_close(total, expected_total)
    torch.testing.assert_close(terms["total"], expected_total)


def test_zero_weight_observable_may_be_omitted():
    loss_fn = EnergyForceTorqueLoss(
        energy_weight=1.0,
        force_weight=0.0,
        torque_weight=0.0,
    )

    total, terms = loss_fn(
        predicted_energy=torch.tensor(2.0),
        target_energy=torch.tensor(1.0),
    )

    torch.testing.assert_close(total, torch.tensor(1.0))
    assert set(terms) == {"energy", "total"}


def test_positive_weight_requires_prediction_and_target():
    loss_fn = EnergyForceTorqueLoss(
        energy_weight=0.0,
        force_weight=1.0,
        torque_weight=0.0,
    )

    try:
        loss_fn(predicted_forces=torch.zeros(1, 3))
    except ValueError as exc:
        assert "force prediction and target are required" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_shape_mismatch_is_rejected():
    loss_fn = EnergyForceTorqueLoss(
        energy_weight=0.0,
        force_weight=1.0,
        torque_weight=0.0,
    )

    try:
        loss_fn(
            predicted_forces=torch.zeros(2, 3),
            target_forces=torch.zeros(3, 3),
        )
    except ValueError as exc:
        assert "force prediction and target shapes differ" in str(exc)
    else:
        raise AssertionError("expected ValueError")


class _ToyConservativeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stiffness = torch.nn.Parameter(
            torch.tensor(1.4, dtype=torch.float64)
        )
        self.orientation_scale = torch.nn.Parameter(
            torch.tensor(0.8, dtype=torch.float64)
        )
        self.direction = torch.nn.Parameter(
            torch.tensor([0.2, -0.7, 0.4], dtype=torch.float64)
        )

    def forward(self, pairs, rotations):
        translational = 0.5 * self.stiffness * pairs.square().sum()
        rotated = rotations @ self.direction
        orientational = self.orientation_scale * rotated[..., 2].sum()
        return translational + orientational


def test_conservative_training_step_reaches_all_parameters():
    model = _ToyConservativeModel()

    pair_vectors = torch.tensor(
        [[0.3, -0.2, 1.1], [-0.4, 0.5, 0.7]],
        dtype=torch.float64,
    )
    base_rotations = torch.eye(
        3,
        dtype=torch.float64,
    ).repeat(2, 1, 1)

    loss_fn = EnergyForceTorqueLoss(
        energy_weight=1.0,
        force_weight=1.0,
        torque_weight=1.0,
    )

    total, terms, predictions = conservative_training_step(
        model,
        pair_vectors,
        base_rotations,
        target_energy=torch.tensor(0.1, dtype=torch.float64),
        target_pair_forces=torch.zeros_like(pair_vectors),
        target_torques=torch.zeros(2, 3, dtype=torch.float64),
        loss_fn=loss_fn,
    )

    energy, pair_forces, torques = predictions
    assert energy.ndim == 0
    assert pair_forces.shape == pair_vectors.shape
    assert torques.shape == (2, 3)
    assert set(terms) == {"energy", "force", "torque", "total"}

    total.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.linalg.vector_norm(parameter.grad) > 0


def test_optimizer_step_reduces_toy_loss():
    model = _ToyConservativeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    pair_vectors = torch.tensor(
        [[0.3, -0.2, 1.1], [-0.4, 0.5, 0.7]],
        dtype=torch.float64,
    )
    base_rotations = torch.eye(
        3,
        dtype=torch.float64,
    ).repeat(2, 1, 1)

    loss_fn = EnergyForceTorqueLoss(
        energy_weight=1.0,
        force_weight=0.2,
        torque_weight=0.2,
    )

    targets = {
        "target_energy": torch.tensor(0.0, dtype=torch.float64),
        "target_pair_forces": torch.zeros_like(pair_vectors),
        "target_torques": torch.zeros(2, 3, dtype=torch.float64),
    }

    before, _, _ = conservative_training_step(
        model,
        pair_vectors,
        base_rotations,
        loss_fn=loss_fn,
        **targets,
    )

    optimizer.zero_grad()
    before.backward()
    optimizer.step()

    after, _, _ = conservative_training_step(
        model,
        pair_vectors,
        base_rotations,
        loss_fn=loss_fn,
        **targets,
    )

    assert after.detach() < before.detach()
