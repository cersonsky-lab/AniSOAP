import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from anisoap.nn import ClebschGordanProduct, select_angular_channels


def _synthetic_coefficients(
    *,
    dtype=torch.float64,
    sample_offset=0,
):
    samples = Labels(
        ["type", "center"],
        torch.tensor(
            [
                [0, sample_offset],
                [0, sample_offset + 1],
            ],
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
            3,
            dtype=dtype,
        )

        blocks.append(
            TensorBlock(
                values=values,
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
                    torch.arange(
                        3,
                        dtype=torch.int32,
                    ).reshape(-1, 1),
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


def _angular_channels(tensor):
    column = list(tensor.keys.names).index("angular_channel")
    return {
        int(row[column])
        for row in tensor.keys.values.detach().cpu().tolist()
    }


def _block_for_l(tensor, l_value):
    column = list(tensor.keys.names).index("angular_channel")

    for block_index, row in enumerate(
        tensor.keys.values.detach().cpu().tolist()
    ):
        if int(row[column]) == l_value:
            return tensor.block(block_index)

    raise KeyError(l_value)


def test_l1_times_l1_obeys_triangle_rule():
    coefficients = select_angular_channels(
        _synthetic_coefficients(),
        active_l=(1,),
    )

    output = ClebschGordanProduct(
        max_angular=2,
        lcut=2,
    )(coefficients, coefficients)

    assert _angular_channels(output) == {0, 1, 2}

    for l_value in (0, 1, 2):
        block = _block_for_l(output, l_value)
        assert block.values.shape[1] == 2 * l_value + 1


def test_lcut_zero_returns_only_scalars():
    coefficients = _synthetic_coefficients()

    output = ClebschGordanProduct(
        max_angular=2,
        lcut=0,
    )(coefficients, coefficients)

    assert _angular_channels(output) == {0}

    for block_index in range(len(output.keys)):
        assert output.block(block_index).values.shape[1] == 1


def test_property_cartesian_product_has_expected_size():
    coefficients = select_angular_channels(
        _synthetic_coefficients(),
        active_l=(1,),
    )

    output = ClebschGordanProduct(
        max_angular=2,
        lcut=0,
    )(coefficients, coefficients)

    scalar = _block_for_l(output, 0)

    assert scalar.values.shape[-1] == 9
    assert len(scalar.properties) == 9


def test_cg_product_preserves_autograd():
    coefficients = _synthetic_coefficients()

    inputs = []
    for block_index in range(len(coefficients.keys)):
        values = coefficients.block(block_index).values
        values.requires_grad_(True)
        inputs.append(values)

    output = ClebschGordanProduct(
        max_angular=2,
        lcut=2,
    )(coefficients, coefficients)

    loss = sum(
        output.block(block_index).values.square().sum()
        for block_index in range(len(output.keys))
    )
    loss.backward()

    for values in inputs:
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()


def test_mismatched_samples_are_rejected():
    left = select_angular_channels(
        _synthetic_coefficients(sample_offset=0),
        active_l=(1,),
    )
    right = select_angular_channels(
        _synthetic_coefficients(sample_offset=10),
        active_l=(1,),
    )

    product = ClebschGordanProduct(
        max_angular=2,
        lcut=2,
    )

    try:
        product(left, right)
    except ValueError as exc:
        assert "matching samples" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched samples")
