import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from anisoap.benchmarks.gb_training import build_parser
from anisoap.nn.energy import LateInvariantEnergyModel


def _coefficients(dtype=torch.float64):
    samples = Labels(
        ["type", "center"],
        torch.tensor([[0, 0], [0, 1]], dtype=torch.int32),
    )
    keys = []
    blocks = []
    for l_value in (0, 2):
        keys.append([0, l_value])
        blocks.append(
            TensorBlock(
                values=torch.randn(
                    2,
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


def test_scalar_features_match_energy_head_input():
    coefficients = _coefficients()
    model = LateInvariantEnergyModel(
        coefficients,
        active_l=(0, 2),
        hidden_channels={0: 4, 2: 4},
        max_angular=2,
    )

    features, samples = model.scalar_features(coefficients)
    center_energies, energy_samples = model.center_energies(coefficients)

    assert samples == energy_samples
    assert features.shape[0] == len(samples)
    assert features.shape[1] == model.energy_head.in_features

    expected = model.energy_head(features).squeeze(-1)
    torch.testing.assert_close(center_energies, expected)


def test_parser_accepts_overfit_mode():
    args = build_parser().parse_args(
        [
            "--overfit-random",
            "16",
            "--overfit-repeats",
            "12",
            "--disable-char-curves",
        ]
    )

    assert args.overfit_random == 16
    assert args.overfit_repeats == 12
    assert args.disable_char_curves
