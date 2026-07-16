"""Late-invariant energy models built from AniSOAP equivariant coefficients."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Tuple

import torch
from metatensor.torch import Labels, TensorMap

from .equivariant import (
    ClebschGordanProduct,
    EquivariantLinear,
    NormGate,
    select_angular_channels,
)


def _scalar_features(tensor: TensorMap) -> Tuple[torch.Tensor, Labels]:
    """Concatenate all ``l=0`` blocks into one feature matrix."""
    key_names = list(tensor.keys.names)
    try:
        angular_index = key_names.index("angular_channel")
    except ValueError as exc:
        raise ValueError(
            "TensorMap keys must contain an 'angular_channel' dimension"
        ) from exc

    features = []
    reference_samples = None

    for block_index, key_row in enumerate(tensor.keys.values.detach().cpu().tolist()):
        if int(key_row[angular_index]) != 0:
            continue

        block = tensor.block(block_index)
        if block.values.shape[1] != 1:
            raise ValueError("l=0 blocks must have one spherical component")

        if reference_samples is None:
            reference_samples = block.samples
        elif block.samples != reference_samples:
            raise ValueError("All scalar blocks must have identical sample labels")

        features.append(block.values[:, 0, :])

    if not features or reference_samples is None:
        raise ValueError("TensorMap does not contain any l=0 blocks")

    return torch.cat(features, dim=-1), reference_samples


def sum_center_energies(
    center_energies: torch.Tensor,
    samples: Labels,
    *,
    system_name: str = "type",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sum center energies by system label.

    The AniSOAP center-descriptor sample labels currently use ``type`` for the
    system/configuration index and ``center`` for the center index.
    """
    if center_energies.ndim != 1:
        raise ValueError("center_energies must be a one-dimensional tensor")
    if len(center_energies) != len(samples):
        raise ValueError(
            "center_energies and samples must contain the same number of rows"
        )
    if system_name not in samples.names:
        raise ValueError(f"samples do not contain the system label {system_name!r}")

    system_column = list(samples.names).index(system_name)
    system_ids = samples.values[:, system_column].to(
        device=center_energies.device,
        dtype=torch.long,
    )
    unique_systems, inverse = torch.unique(
        system_ids,
        sorted=True,
        return_inverse=True,
    )

    energies = torch.zeros(
        len(unique_systems),
        dtype=center_energies.dtype,
        device=center_energies.device,
    )
    energies.index_add_(0, inverse, center_energies)
    return energies, unique_systems


class LateInvariantEnergyModel(torch.nn.Module):
    """One-interaction late-invariant energy model.

    The model consumes the equivariant coefficient ``TensorMap`` returned by
    ``EllipsoidalDensityProjection.transform()``. It selects angular channels,
    applies learned channel mixing and an equivariant norm gate, takes a
    Clebsch--Gordan self-product, and contracts to ``L=0`` only immediately
    before the scalar energy head.
    """

    def __init__(
        self,
        example: TensorMap,
        *,
        active_l: Iterable[int],
        hidden_channels: int | Mapping[int, int],
        max_angular: int,
        system_name: str = "type",
        scalar_hidden: int = 0,
    ) -> None:
        super().__init__()
        self.active_l = tuple(int(value) for value in active_l)
        if not self.active_l:
            raise ValueError("active_l must not be empty")

        self.system_name = system_name
        selected = select_angular_channels(example, self.active_l)
        self.channel_mixing = EquivariantLinear(
            selected,
            hidden_channels,
        )
        mixed = self.channel_mixing(selected)
        self.gate = NormGate(mixed)
        gated = self.gate(mixed)
        self.product = ClebschGordanProduct(
            max_angular=max_angular,
            lcut=0,
        )
        scalar_map = self.product(gated, gated)
        scalar_features, _ = _scalar_features(scalar_map)

        if scalar_hidden < 0:
            raise ValueError("scalar_hidden must be non-negative")
        if scalar_hidden == 0:
            self.energy_head = torch.nn.Linear(
                scalar_features.shape[-1],
                1,
                dtype=scalar_features.dtype,
                device=scalar_features.device,
            )
        else:
            self.energy_head = torch.nn.Sequential(
                torch.nn.Linear(
                    scalar_features.shape[-1],
                    scalar_hidden,
                    dtype=scalar_features.dtype,
                    device=scalar_features.device,
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(
                    scalar_hidden,
                    scalar_hidden,
                    dtype=scalar_features.dtype,
                    device=scalar_features.device,
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(
                    scalar_hidden,
                    1,
                    dtype=scalar_features.dtype,
                    device=scalar_features.device,
                ),
            )

    def scalar_features(
        self,
        coefficients: TensorMap,
    ) -> Tuple[torch.Tensor, Labels]:
        """Return the final invariant features before the energy head."""
        selected = select_angular_channels(
            coefficients,
            self.active_l,
        )
        mixed = self.channel_mixing(selected)
        gated = self.gate(mixed)
        scalar_map = self.product(gated, gated)
        return _scalar_features(scalar_map)

    def center_energies(
        self,
        coefficients: TensorMap,
    ) -> Tuple[torch.Tensor, Labels]:
        scalar_features, samples = self.scalar_features(coefficients)
        energies = self.energy_head(scalar_features).squeeze(-1)
        return energies, samples

    def forward(self, coefficients: TensorMap) -> torch.Tensor:
        center_energies, samples = self.center_energies(coefficients)
        system_energies, _ = sum_center_energies(
            center_energies,
            samples,
            system_name=self.system_name,
        )
        return system_energies
