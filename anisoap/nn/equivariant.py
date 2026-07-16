"""Equivariant operations on AniSOAP coefficient ``TensorMap`` objects.

These layers operate on the output of
``EllipsoidalDensityProjection.transform()``. They do not contract spherical
components to rotational invariants.

Conventions
-----------
Each block is expected to have:

- one spherical-component axis of length ``2 * l + 1``;
- properties/channels on the final axis;
- an ``angular_channel`` entry in the block key.

Learned linear maps act only on the property axis. The spherical component
axis is never mixed, which preserves SO(3) equivariance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import List, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from anisoap.utils.metatensor_utils import (
    TorchClebschGordanReal,
    cg_combine,
    standardize_keys,
)


def _key_rows(keys: Labels) -> List[Tuple[int, ...]]:
    """Convert label rows to plain integer tuples."""
    return [
        tuple(int(value) for value in row)
        for row in keys.values.detach().cpu().tolist()
    ]


def _angular_channel(key_row: Tuple[int, ...], key_names: List[str]) -> int:
    try:
        index = key_names.index("angular_channel")
    except ValueError as exc:
        raise ValueError(
            "TensorMap keys must contain an 'angular_channel' dimension"
        ) from exc
    return int(key_row[index])


def _signature(tensor: TensorMap) -> Tuple[Tuple[int, ...], ...]:
    return tuple(_key_rows(tensor.keys))


def select_angular_channels(
    tensor: TensorMap,
    active_l: Iterable[int],
) -> TensorMap:
    """Select coefficient blocks with angular channels in ``active_l``.

    Parameters
    ----------
    tensor
        AniSOAP coefficient ``TensorMap``, normally returned by
        ``EllipsoidalDensityProjection.transform()``.
    active_l
        Angular channels to retain.

    Returns
    -------
    TensorMap
        A new map reusing the selected blocks.
    """
    selected = {int(l_value) for l_value in active_l}
    if not selected:
        raise ValueError("active_l must contain at least one angular channel")

    key_names = list(tensor.keys.names)
    key_rows = _key_rows(tensor.keys)

    output_keys = []
    output_blocks = []

    for block_index, key_row in enumerate(key_rows):
        if _angular_channel(key_row, key_names) in selected:
            block = tensor.block(block_index)
            output_keys.append(key_row)
            output_blocks.append(
                TensorBlock(
                    values=block.values,
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
            )

    if not output_blocks:
        raise ValueError(
            f"No coefficient blocks found for angular channels {sorted(selected)}"
        )

    key_values = torch.tensor(
        output_keys,
        dtype=tensor.keys.values.dtype,
        device=tensor.keys.values.device,
    )

    return TensorMap(
        keys=Labels(tensor.keys.names, key_values),
        blocks=output_blocks,
    )


class EquivariantLinear(torch.nn.Module):
    """Mix property channels without mixing spherical components.

    One weight matrix is learned per TensorMap block. A bias is permitted only
    for ``l = 0`` because adding a constant to an ``l > 0`` irrep would break
    rotational equivariance.
    """

    def __init__(
        self,
        example: TensorMap,
        out_channels: int | Mapping[int, int],
        *,
        bias_l0: bool = True,
    ) -> None:
        super().__init__()

        self._signature = _signature(example)
        self._key_names = list(example.keys.names)
        self._output_sizes: List[int] = []
        self._use_bias: List[bool] = []

        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()

        for block_index, key_row in enumerate(_key_rows(example.keys)):
            block = example.block(block_index)
            l_value = _angular_channel(key_row, self._key_names)

            if isinstance(out_channels, Mapping):
                if l_value not in out_channels:
                    raise ValueError(
                        f"No output channel count supplied for l={l_value}"
                    )
                n_output = int(out_channels[l_value])
            else:
                n_output = int(out_channels)

            if n_output <= 0:
                raise ValueError("out_channels must be positive")

            n_input = int(block.values.shape[-1])
            weight = torch.empty(
                (n_output, n_input),
                dtype=block.values.dtype,
                device=block.values.device,
            )
            torch.nn.init.orthogonal_(weight)
            self.weights.append(torch.nn.Parameter(weight))

            use_bias = bool(bias_l0 and l_value == 0)
            bias = torch.zeros(
                n_output,
                dtype=block.values.dtype,
                device=block.values.device,
            )
            self.biases.append(torch.nn.Parameter(bias, requires_grad=use_bias))

            self._output_sizes.append(n_output)
            self._use_bias.append(use_bias)

    def forward(self, tensor: TensorMap) -> TensorMap:
        if _signature(tensor) != self._signature:
            raise ValueError(
                "Input TensorMap keys differ from the example used to "
                "construct EquivariantLinear"
            )

        output_blocks = []

        for block_index in range(len(tensor.keys)):
            block = tensor.block(block_index)
            weight = self.weights[block_index]

            values = torch.einsum(
                "...p,op->...o",
                block.values,
                weight,
            )

            if self._use_bias[block_index]:
                values = values + self.biases[block_index]

            properties = Labels(
                ["channel"],
                torch.arange(
                    self._output_sizes[block_index],
                    dtype=torch.int32,
                    device=values.device,
                ).reshape(-1, 1),
            )

            output_blocks.append(
                TensorBlock(
                    values=values,
                    samples=block.samples,
                    components=block.components,
                    properties=properties,
                )
            )

        return TensorMap(keys=tensor.keys, blocks=output_blocks)


class NormGate(torch.nn.Module):
    """Apply an equivariant nonlinearity using irrep norms.

    For each sample and property channel, this computes a scalar gate from

    ``sqrt(sum_m x_m**2 + epsilon)``

    and multiplies every spherical component by the same scalar. Because the
    norm is rotationally invariant, this operation preserves equivariance.
    """

    def __init__(
        self,
        example: TensorMap,
        *,
        epsilon: float = 1.0e-12,
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.epsilon = float(epsilon)
        self._signature = _signature(example)

        self.alpha = torch.nn.ParameterList()
        self.beta = torch.nn.ParameterList()

        for block_index in range(len(example.keys)):
            block = example.block(block_index)
            if len(block.components) != 1:
                raise ValueError(
                    "NormGate expects exactly one spherical-component axis"
                )

            channels = int(block.values.shape[-1])
            self.alpha.append(
                torch.nn.Parameter(
                    torch.ones(
                        channels,
                        dtype=block.values.dtype,
                        device=block.values.device,
                    )
                )
            )
            self.beta.append(
                torch.nn.Parameter(
                    torch.zeros(
                        channels,
                        dtype=block.values.dtype,
                        device=block.values.device,
                    )
                )
            )

    def forward(self, tensor: TensorMap) -> TensorMap:
        if _signature(tensor) != self._signature:
            raise ValueError(
                "Input TensorMap keys differ from the example used to "
                "construct NormGate"
            )

        output_blocks = []

        for block_index in range(len(tensor.keys)):
            block = tensor.block(block_index)

            if len(block.components) != 1:
                raise ValueError(
                    "NormGate expects exactly one spherical-component axis"
                )

            norm = torch.sqrt(
                torch.sum(
                    block.values.square(),
                    dim=1,
                    keepdim=True,
                )
                + self.epsilon
            )
            gate = torch.sigmoid(
                self.alpha[block_index] * norm + self.beta[block_index]
            )

            output_blocks.append(
                TensorBlock(
                    values=block.values * gate,
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
            )

        return TensorMap(keys=tensor.keys, blocks=output_blocks)


class ClebschGordanProduct(torch.nn.Module):
    """Combine two equivariant AniSOAP coefficient maps.

    This wraps AniSOAP's existing real Clebsch--Gordan implementation, so it
    uses the same spherical-harmonic convention as the rest of the package.

    Parameters
    ----------
    max_angular
        Largest angular channel needed by the input or output.
    lcut
        Largest output angular channel. Defaults to ``max_angular``.
    other_keys_match
        Non-angular key dimensions that must agree in the two inputs.
    """

    def __init__(
        self,
        max_angular: int,
        *,
        lcut: int | None = None,
        other_keys_match: Tuple[str, ...] = ("types_center",),
    ) -> None:
        super().__init__()

        self.max_angular = int(max_angular)
        self.lcut = self.max_angular if lcut is None else int(lcut)
        self.other_keys_match = tuple(other_keys_match)

        if self.max_angular < 0:
            raise ValueError("max_angular must be non-negative")
        if self.lcut < 0:
            raise ValueError("lcut must be non-negative")
        if self.lcut > self.max_angular:
            raise ValueError(
                "lcut can not exceed max_angular because the required "
                "Clebsch-Gordan tables would not be available"
            )

        self._clebsch_gordan = TorchClebschGordanReal(self.max_angular)

    def forward(
        self,
        left: TensorMap,
        right: TensorMap,
    ) -> TensorMap:
        """Return the real Clebsch--Gordan product."""
        return cg_combine(
            standardize_keys(left),
            standardize_keys(right),
            clebsch_gordan=self._clebsch_gordan,
            lcut=self.lcut,
            other_keys_match=self.other_keys_match,
        )
