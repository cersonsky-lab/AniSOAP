import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from anisoap.nn import LateInvariantEnergyModel, sum_center_energies


def _coefficients(dtype=torch.float64):
    samples = Labels(
        ["type", "center"],
        torch.tensor(
            [[0, 0], [0, 1], [1, 0]],
            dtype=torch.int32,
        ),
    )
    keys = []
    blocks = []
    for l_value in (0, 1, 2):
        keys.append([0, l_value])
        blocks.append(
            TensorBlock(
                values=torch.randn(
                    3,
                    2 * l_value + 1,
                    3,
                    dtype=dtype,
                ),
                samples=samples,
                components=[
                    Labels(
                        ["spherical_component_m"],
                        torch.arange(
                            -l_value,
                            l_value + 1,
                            dtype=torch.int32,
                        ).reshape(-1, 1),
                    )
                ],
                properties=Labels(
                    ["channel"],
                    torch.arange(3, dtype=torch.int32).reshape(-1, 1),
                ),
            )
        )
    return TensorMap(
        keys=Labels(
            ["types_center", "angular_channel"],
            torch.tensor(keys, dtype=torch.int32),
        ),
        blocks=blocks,
    )


def test_sum_center_energies():
    samples = Labels(
        ["type", "center"],
        torch.tensor(
            [[0, 0], [0, 1], [1, 0]],
            dtype=torch.int32,
        ),
    )
    totals, systems = sum_center_energies(
        torch.tensor([1.0, 2.0, 4.0]),
        samples,
    )
    torch.testing.assert_close(totals, torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(systems, torch.tensor([0, 1]))


def test_model_returns_one_energy_per_system():
    coefficients = _coefficients()
    model = LateInvariantEnergyModel(
        coefficients,
        active_l=(0, 1, 2),
        hidden_channels=2,
        max_angular=2,
    )
    energies = model(coefficients)
    assert energies.shape == (2,)
    assert torch.isfinite(energies).all()


def test_center_energies_keep_center_samples():
    coefficients = _coefficients()
    model = LateInvariantEnergyModel(
        coefficients,
        active_l=(0, 1),
        hidden_channels={0: 2, 1: 3},
        max_angular=2,
    )
    center_energies, samples = model.center_energies(coefficients)
    assert center_energies.shape == (3,)
    assert samples == coefficients.block(0).samples


def test_full_model_preserves_autograd():
    coefficients = _coefficients()
    inputs = []
    for block_index in range(len(coefficients.keys)):
        values = coefficients.block(block_index).values
        values.requires_grad_(True)
        inputs.append(values)

    model = LateInvariantEnergyModel(
        coefficients,
        active_l=(0, 1, 2),
        hidden_channels=2,
        max_angular=2,
    )
    loss = model(coefficients).square().sum()
    loss.backward()

    for values in inputs:
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()

    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_energy_changes_when_equivariant_coefficients_change():
    coefficients = _coefficients()
    model = LateInvariantEnergyModel(
        coefficients,
        active_l=(0, 1, 2),
        hidden_channels=2,
        max_angular=2,
    )

    before = model(coefficients).detach().clone()
    coefficients.block(2).values[0, 0, 0] += 0.5
    after = model(coefficients).detach()

    assert not torch.allclose(before, after)
