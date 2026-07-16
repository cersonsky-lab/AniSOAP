"""Losses and helpers for conservative energy/force/torque fitting."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

from .mechanics import conservative_pair_forces_and_torques


class EnergyForceTorqueLoss(torch.nn.Module):
    """Weighted mean-squared error for energy, force, and torque targets.

    Any observable with zero weight may be omitted. An observable with a
    positive weight must have both a prediction and a target.

    Parameters
    ----------
    energy_weight
        Weight multiplying the energy mean-squared error.
    force_weight
        Weight multiplying the force mean-squared error.
    torque_weight
        Weight multiplying the torque mean-squared error.
    """

    def __init__(
        self,
        *,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        torque_weight: float = 1.0,
    ) -> None:
        super().__init__()

        weights = {
            "energy": float(energy_weight),
            "force": float(force_weight),
            "torque": float(torque_weight),
        }
        for name, value in weights.items():
            if value < 0:
                raise ValueError(f"{name}_weight must be non-negative")

        if all(value == 0 for value in weights.values()):
            raise ValueError("at least one observable weight must be positive")

        self.energy_weight = weights["energy"]
        self.force_weight = weights["force"]
        self.torque_weight = weights["torque"]

    @staticmethod
    def _mse(
        prediction: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        *,
        name: str,
        required: bool,
    ) -> Optional[torch.Tensor]:
        if not required:
            return None

        if prediction is None or target is None:
            raise ValueError(
                f"{name} prediction and target are required when its "
                "loss weight is positive"
            )
        if prediction.shape != target.shape:
            raise ValueError(
                f"{name} prediction and target shapes differ: "
                f"{tuple(prediction.shape)} != {tuple(target.shape)}"
            )

        return torch.mean((prediction - target).square())

    def forward(
        self,
        *,
        predicted_energy: Optional[torch.Tensor] = None,
        target_energy: Optional[torch.Tensor] = None,
        predicted_forces: Optional[torch.Tensor] = None,
        target_forces: Optional[torch.Tensor] = None,
        predicted_torques: Optional[torch.Tensor] = None,
        target_torques: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return total loss and unweighted per-observable MSE terms."""
        terms: Dict[str, torch.Tensor] = {}
        weighted_terms = []

        energy = self._mse(
            predicted_energy,
            target_energy,
            name="energy",
            required=self.energy_weight > 0,
        )
        if energy is not None:
            terms["energy"] = energy
            weighted_terms.append(self.energy_weight * energy)

        force = self._mse(
            predicted_forces,
            target_forces,
            name="force",
            required=self.force_weight > 0,
        )
        if force is not None:
            terms["force"] = force
            weighted_terms.append(self.force_weight * force)

        torque = self._mse(
            predicted_torques,
            target_torques,
            name="torque",
            required=self.torque_weight > 0,
        )
        if torque is not None:
            terms["torque"] = torque
            weighted_terms.append(self.torque_weight * torque)

        total = torch.stack(weighted_terms).sum()
        terms["total"] = total
        return total, terms


def conservative_training_step(
    energy_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pair_vectors: torch.Tensor,
    base_rotations: torch.Tensor,
    *,
    target_energy: Optional[torch.Tensor],
    target_pair_forces: Optional[torch.Tensor],
    target_torques: Optional[torch.Tensor],
    loss_fn: EnergyForceTorqueLoss,
) -> Tuple[
    torch.Tensor,
    Dict[str, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Evaluate conservative observables and their combined training loss.

    ``create_graph=True`` is always used so force and torque residuals can
    backpropagate through the derivative calculation to model parameters.

    This helper operates on one graph or one already-assembled minibatch.
    ``energy_fn`` may return multiple system energies; the mechanics helper
    differentiates their sum and returns that same scalar total energy.
    """
    energy, pair_forces, torques = conservative_pair_forces_and_torques(
        energy_fn,
        pair_vectors,
        base_rotations,
        create_graph=True,
    )

    total, terms = loss_fn(
        predicted_energy=energy,
        target_energy=target_energy,
        predicted_forces=pair_forces,
        target_forces=target_pair_forces,
        predicted_torques=torques,
        target_torques=target_torques,
    )

    return total, terms, (energy, pair_forces, torques)
