import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from anisoap.nn import EquivariantLinear, NormGate, select_angular_channels


def _synthetic_coefficients(dtype=torch.float64):
    samples = Labels(
        ["type", "center"],
        torch.tensor(
            [[0, 0], [0, 1]],
            dtype=torch.int32,
        ),
    )

    keys = []
    blocks = []

    for l_value in (0, 1, 2):
        keys.append([0, l_value])

        values = torch.randn(
            2,
            2 * l_value + 1,
            4,
            dtype=dtype,
        )

        component = Labels(
            ["spherical_component_m"],
            torch.arange(
                -l_value,
                l_value + 1,
                dtype=torch.int32,
            ).reshape(-1, 1),
        )

        properties = Labels(
            ["n", "neighbor_type"],
            torch.tensor(
                [[0, 0], [1, 0], [2, 0], [3, 0]],
                dtype=torch.int32,
            ),
        )

        blocks.append(
            TensorBlock(
                values=values,
                samples=samples,
                components=[component],
                properties=properties,
            )
        )

    return TensorMap(
        keys=Labels(
            ["types_center", "angular_channel"],
            torch.tensor(keys, dtype=torch.int32),
        ),
        blocks=blocks,
    )


def _angular_channels(tensor):
    column = list(tensor.keys.names).index("angular_channel")
    return {
        int(row[column])
        for row in tensor.keys.values.detach().cpu().tolist()
    }


def test_select_angular_channels():
    coefficients = _synthetic_coefficients()

    selected = select_angular_channels(
        coefficients,
        active_l=(0, 2),
    )

    assert len(selected.keys) == 2
    assert _angular_channels(selected) == {0, 2}


def test_select_angular_channels_rejects_missing_selection():
    coefficients = _synthetic_coefficients()

    try:
        select_angular_channels(coefficients, active_l=(4,))
    except ValueError as exc:
        assert "No coefficient blocks" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_equivariant_linear_only_changes_property_dimension():
    coefficients = _synthetic_coefficients()

    layer = EquivariantLinear(
        coefficients,
        out_channels={0: 3, 1: 5, 2: 7},
    )
    output = layer(coefficients)

    for block_index, key_row in enumerate(
        output.keys.values.detach().cpu().tolist()
    ):
        l_value = int(key_row[1])
        block = output.block(block_index)

        assert block.values.shape == (
            2,
            2 * l_value + 1,
            {0: 3, 1: 5, 2: 7}[l_value],
        )
        assert block.samples == coefficients.block(block_index).samples
        assert block.components == coefficients.block(block_index).components


def test_bias_is_restricted_to_l_zero():
    coefficients = _synthetic_coefficients()
    layer = EquivariantLinear(coefficients, out_channels=3)

    with torch.no_grad():
        for weight in layer.weights:
            weight.zero_()
        for bias in layer.biases:
            bias.fill_(2.0)

    output = layer(coefficients)

    for block_index, key_row in enumerate(
        output.keys.values.detach().cpu().tolist()
    ):
        l_value = int(key_row[1])
        values = output.block(block_index).values

        if l_value == 0:
            torch.testing.assert_close(
                values,
                torch.full_like(values, 2.0),
            )
        else:
            torch.testing.assert_close(
                values,
                torch.zeros_like(values),
            )


def test_norm_gate_uses_same_scalar_for_every_m_component():
    coefficients = _synthetic_coefficients()

    # Avoid division by zero in the ratio check.
    for block_index in range(len(coefficients.keys)):
        block = coefficients.block(block_index)
        block.values[:] = block.values.abs() + 0.2

    output = NormGate(coefficients)(coefficients)

    for block_index in range(len(coefficients.keys)):
        before = coefficients.block(block_index).values
        after = output.block(block_index).values

        ratio = after / before
        reference = ratio[:, :1, :]

        torch.testing.assert_close(
            ratio,
            reference.expand_as(ratio),
        )


def test_layers_preserve_autograd():
    coefficients = _synthetic_coefficients()

    inputs = []
    for block_index in range(len(coefficients.keys)):
        block = coefficients.block(block_index)
        block.values.requires_grad_(True)
        inputs.append(block.values)

    mixed = EquivariantLinear(
        coefficients,
        out_channels=3,
    )(coefficients)
    gated = NormGate(mixed)(mixed)

    loss = sum(
        gated.block(block_index).values.square().sum()
        for block_index in range(len(gated.keys))
    )
    loss.backward()

    for values in inputs:
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()



def test_selected_blocks_preserve_values_and_autograd():
    coefficients = _synthetic_coefficients()

    original_by_l = {}

    for block_index, key_row in enumerate(
        coefficients.keys.values.detach().cpu().tolist()
    ):
        l_value = int(key_row[1])
        block = coefficients.block(block_index)
        block.values.requires_grad_(True)
        original_by_l[l_value] = block

    selected = select_angular_channels(
        coefficients,
        active_l=(0, 2),
    )

    assert _angular_channels(selected) == {0, 2}

    for block_index, key_row in enumerate(
        selected.keys.values.detach().cpu().tolist()
    ):
        l_value = int(key_row[1])
        selected_block = selected.block(block_index)
        original_block = original_by_l[l_value]

        torch.testing.assert_close(
            selected_block.values,
            original_block.values,
        )
        assert selected_block.samples == original_block.samples
        assert selected_block.components == original_block.components
        assert selected_block.properties == original_block.properties

    loss = sum(
        selected.block(block_index).values.square().sum()
        for block_index in range(len(selected.keys))
    )
    loss.backward()

    assert original_by_l[0].values.grad is not None
    assert original_by_l[2].values.grad is not None
    assert original_by_l[1].values.grad is None
