"""Conservative force and torque extraction from scalar energy models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Tuple

import torch


def rotation_vector_to_matrix(rotation_vectors: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotation vectors to rotation matrices.

    Parameters
    ----------
    rotation_vectors
        Tensor with shape ``(..., 3)``. Its direction is the rotation axis and
        its norm is the rotation angle in radians.

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape ``(..., 3, 3)``.

    Notes
    -----
    The implementation uses Rodrigues' formula with small-angle series, and
    is differentiable at zero rotation. This is important when torques are
    evaluated as derivatives at an unperturbed configuration.
    """
    if rotation_vectors.shape[-1] != 3:
        raise ValueError("rotation_vectors must have shape (..., 3)")

    x, y, z = rotation_vectors.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ),
        dim=-1,
    ).reshape(rotation_vectors.shape[:-1] + (3, 3))

    # ``torch.matrix_exp`` evaluates the exponential map directly. Unlike a
    # hand-written Rodrigues formula, it has no removable 0/0 singularities,
    # and its backward pass is finite at the zero rotation vector.
    return torch.matrix_exp(skew)


def apply_space_rotations(
    base_rotations: torch.Tensor,
    rotation_vectors: torch.Tensor,
) -> torch.Tensor:
    """Apply space-frame perturbations to existing rotation matrices.

    The convention is

    ``R_new = Exp(delta_theta) @ R_base``.

    Therefore ``-dE/d(delta_theta)`` at ``delta_theta = 0`` is the torque in
    the space/laboratory frame.
    """
    if base_rotations.shape[-2:] != (3, 3):
        raise ValueError("base_rotations must have shape (..., 3, 3)")
    if rotation_vectors.shape != base_rotations.shape[:-2] + (3,):
        raise ValueError(
            "rotation_vectors shape must match base_rotations batch dimensions"
        )

    return rotation_vector_to_matrix(rotation_vectors) @ base_rotations


def conservative_pair_forces_and_torques(
    energy_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pair_vectors: torch.Tensor,
    base_rotations: torch.Tensor,
    *,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate a scalar energy to obtain pair forces and torques.

    Parameters
    ----------
    energy_fn
        Callable taking ``(pair_vectors, rotation_matrices)`` and returning one
        or more system energies. Energies are summed before differentiation.
    pair_vectors
        Pair displacement tensor, usually AniSOAP's ``graph.R_ij``.
    base_rotations
        Particle rotation matrices at the evaluation configuration.
    create_graph
        Keep the derivative graph, as required when force/torque residuals are
        used in model training.

    Returns
    -------
    energy, pair_forces, torques
        ``pair_forces = -dE/dR_ij``. Torques contain three physical components
        per rotation and use the space-frame convention documented in
        :func:`apply_space_rotations`.
    """
    if pair_vectors.shape[-1] != 3:
        raise ValueError("pair_vectors must have shape (..., 3)")
    if base_rotations.shape[-2:] != (3, 3):
        raise ValueError("base_rotations must have shape (..., 3, 3)")

    if pair_vectors.requires_grad:
        differentiable_pairs = pair_vectors
    else:
        differentiable_pairs = pair_vectors.detach().clone().requires_grad_(True)

    rotation_vectors = torch.zeros(
        base_rotations.shape[:-2] + (3,),
        dtype=base_rotations.dtype,
        device=base_rotations.device,
        requires_grad=True,
    )
    rotations = apply_space_rotations(base_rotations, rotation_vectors)

    energies = energy_fn(differentiable_pairs, rotations)
    if not isinstance(energies, torch.Tensor):
        raise TypeError("energy_fn must return a torch.Tensor")
    energy = energies.sum()

    pair_gradient, rotation_gradient = torch.autograd.grad(
        energy,
        (differentiable_pairs, rotation_vectors),
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=True,
    )

    if pair_gradient is None:
        pair_gradient = torch.zeros_like(differentiable_pairs)
    if rotation_gradient is None:
        rotation_gradient = torch.zeros_like(rotation_vectors)

    return energy, -pair_gradient, -rotation_gradient


def pair_forces_to_atom_forces(
    pair_forces: torch.Tensor,
    centers: torch.Tensor,
    neighbors: torch.Tensor,
    *,
    n_atoms: int,
) -> torch.Tensor:
    """Scatter pair forces to atoms while enforcing Newton's third law.

    ``pair_forces[p]`` is interpreted as the force on the neighbor of pair
    ``p`` for ``R_ij = r_j - r_i``. The center receives the opposite force.
    """
    if pair_forces.ndim != 2 or pair_forces.shape[-1] != 3:
        raise ValueError("pair_forces must have shape (n_pairs, 3)")
    if centers.ndim != 1 or neighbors.ndim != 1:
        raise ValueError("centers and neighbors must be one-dimensional")
    if len(pair_forces) != len(centers) or len(pair_forces) != len(neighbors):
        raise ValueError("pair forces and pair indices must have equal lengths")
    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive")

    centers = centers.to(device=pair_forces.device, dtype=torch.long)
    neighbors = neighbors.to(device=pair_forces.device, dtype=torch.long)

    atom_forces = torch.zeros(
        (n_atoms, 3),
        dtype=pair_forces.dtype,
        device=pair_forces.device,
    )
    atom_forces.index_add_(0, centers, -pair_forces)
    atom_forces.index_add_(0, neighbors, pair_forces)
    return atom_forces
