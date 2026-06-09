"""Torch-native ellipsoidal density projection for AniSOAP."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
from anisoap.representations.radial_basis import (
    GTORadialBasis,
    MonomialBasis,
    _RadialBasis,
)
from anisoap.utils.spherical_to_cartesian import spherical_to_cartesian
from metatensor.torch import (
    Labels,
    TensorBlock,
    TensorMap,
)

from ..utils.metatensor_utils import (
    TorchClebschGordanReal,
    _key_value,
    _labels_to_tuples,
    cg_combine,
    standardize_keys,
)
from .radial_basis import (
    gaussian_parameters,
    orthonormalization_matrix,
)


def _moment_index_maps(maxdeg: int, device=None):
    """Build compact monomial maps for all exponents with total degree <= maxdeg."""
    exponents_list: List[Tuple[int, int, int]] = []
    for degree in range(maxdeg + 1):
        for n0 in range(degree + 1):
            for n1 in range(degree + 1 - n0):
                n2 = degree - n0 - n1
                exponents_list.append((n0, n1, n2))

    index = {exp: i for i, exp in enumerate(exponents_list)}
    exponents = torch.tensor(exponents_list, device=device, dtype=torch.long)
    degrees = exponents.sum(dim=1)

    parent = torch.full((len(exponents_list),), -1, device=device, dtype=torch.long)
    direction = torch.full((len(exponents_list),), -1, device=device, dtype=torch.long)
    decrement = torch.full(
        (len(exponents_list), 3), -1, device=device, dtype=torch.long
    )

    for i, (n0, n1, n2) in enumerate(exponents_list):
        if n0 + n1 + n2 == 0:
            continue
        if n0 > 0:
            k = 0
            p = (n0 - 1, n1, n2)
        elif n1 > 0:
            k = 1
            p = (n0, n1 - 1, n2)
        else:
            k = 2
            p = (n0, n1, n2 - 1)

        parent[i] = index[p]
        direction[i] = k
        p_list = list(p)
        for j in range(3):
            if p_list[j] > 0:
                q = p_list.copy()
                q[j] -= 1
                decrement[i, j] = index[tuple(q)]

    return exponents, degrees, parent, direction, decrement


def compute_moments_batched(
    A: torch.Tensor,
    a: torch.Tensor,
    maxdeg: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Batched unnormalized trivariate Gaussian raw moments.

    Computes moments of ``exp(-1/2 (x-a)^T A (x-a))`` up to total polynomial
    degree ``maxdeg``. The result is compact: only exponents with total degree
    <= maxdeg are stored.

    Returns
    -------
    moments
        Shape ``(batch, n_monomials)``.
    exponents
        Shape ``(n_monomials, 3)``; each row is ``(n0, n1, n2)``.
    """
    A = torch.as_tensor(A)
    if A.ndim == 2:
        A = A.reshape(1, 3, 3)
    a = torch.as_tensor(a, device=A.device, dtype=A.dtype)
    if a.ndim == 1:
        a = a.reshape(1, 3)
    if A.shape[-2:] != (3, 3):
        raise ValueError(f"A must have shape (..., 3, 3), got {tuple(A.shape)}")
    if a.shape[-1] != 3:
        raise ValueError(f"a must have shape (..., 3), got {tuple(a.shape)}")
    if A.shape[0] != a.shape[0]:
        raise ValueError("A and a must have the same batch dimension")
    if maxdeg < 0:
        raise ValueError("maxdeg must be non-negative")

    device = A.device
    dtype = A.dtype
    batch = A.shape[0]
    exponents, degrees, parent, direction, decrement = _moment_index_maps(
        maxdeg, device=device
    )

    cov = torch.linalg.inv(A)
    sign, logabsdet = torch.linalg.slogdet(A)
    if not bool((sign > 0).all()):
        raise ValueError("Gaussian precision matrices must be positive definite")
    norm = torch.exp(1.5 * math.log(2.0 * math.pi) - 0.5 * logabsdet)

    moments = torch.zeros((batch, exponents.shape[0]), device=device, dtype=dtype)
    moments[:, 0] = 1.0

    for degree in range(1, maxdeg + 1):
        ids = torch.nonzero(degrees == degree, as_tuple=False).reshape(-1)
        p = parent[ids]
        k = direction[ids]
        values = a[:, k] * moments[:, p]
        parent_exponents = exponents[p]

        for j in range(3):
            dec = decrement[ids, j]
            valid = dec >= 0
            if bool(valid.any()):
                coeff = parent_exponents[valid, j].to(dtype=dtype)
                values[:, valid] = values[:, valid] + (
                    coeff.reshape(1, -1) * cov[:, k[valid], j] * moments[:, dec[valid]]
                )

        moments[:, ids] = values

    return norm.reshape(-1, 1) * moments, exponents


def _compact_moments_to_cube(
    moments: torch.Tensor,
    exponents: torch.Tensor,
    maxdeg: int,
) -> torch.Tensor:
    cube = torch.zeros(
        (moments.shape[0], maxdeg + 1, maxdeg + 1, maxdeg + 1),
        device=moments.device,
        dtype=moments.dtype,
    )
    cube[:, exponents[:, 0], exponents[:, 1], exponents[:, 2]] = moments
    return cube


def compute_moments(A: torch.Tensor, a: torch.Tensor, maxdeg: int) -> torch.Tensor:
    r"""Compatibility wrapper returning the historical dense moment cube."""
    moments, exponents = compute_moments_batched(A, a, maxdeg)
    return _compact_moments_to_cube(moments, exponents, maxdeg)[0]


@dataclass
class AniSOAPGraph:
    R_ij: torch.Tensor
    centers: torch.Tensor
    neighbors: torch.Tensor
    species: torch.Tensor
    structures: torch.Tensor
    atom_indices: torch.Tensor
    rotations: torch.Tensor
    ellipsoid_lengths: torch.Tensor


def _get_system_tensor_data(system: Any, name: str) -> torch.Tensor:
    """Read per-atom custom data from a metatomic System.

    Supports both direct tensor attributes and TensorMap/TensorBlock-style data.
    This is deliberately only for AniSOAP's custom per-atom fields, not for
    neighbor-list TensorMaps, which are consumed as-is.
    """
    if hasattr(system, name):
        data = getattr(system, name)
    elif hasattr(system, "get_data"):
        data = system.get_data(name)
    else:
        raise ValueError(f"System does not contain required AniSOAP data '{name}'")

    if torch.is_tensor(data):
        return data
    if isinstance(data, TensorMap):
        return data.block(0).values
    if isinstance(data, TensorBlock):
        return data.values
    raise TypeError(f"Unsupported storage for System data '{name}': {type(data)!r}")


def _extract_neighbor_list_edges(
    system: Any, options: Any, system_index: int, atom_offset: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract global center/neighbor indices and pair vectors from a System neighbor list."""
    nl = system.get_neighbor_list(options)
    centers_all: List[torch.Tensor] = []
    neighbors_all: List[torch.Tensor] = []
    vectors_all: List[torch.Tensor] = []

    for _, block in nl.items():
        samples = block.samples
        sample_names = list(samples.names)
        sample_values = samples.values.to(dtype=torch.long, device=block.values.device)
        if "first_atom" not in sample_names or "second_atom" not in sample_names:
            raise ValueError(
                "Neighbor list samples must contain 'first_atom' and 'second_atom'"
            )
        first = sample_values[:, sample_names.index("first_atom")]
        second = sample_values[:, sample_names.index("second_atom")]
        values = block.values
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("Neighbor list block values must have shape (n_pairs, 3)")
        centers_all.append(first + atom_offset)
        neighbors_all.append(second + atom_offset)
        vectors_all.append(values)

    if vectors_all:
        return (
            torch.cat(vectors_all, dim=0),
            torch.cat(centers_all, dim=0),
            torch.cat(neighbors_all, dim=0),
        )

    device = system.positions.device
    dtype = system.positions.dtype
    return (
        torch.empty((0, 3), device=device, dtype=dtype),
        torch.empty((0,), device=device, dtype=torch.long),
        torch.empty((0,), device=device, dtype=torch.long),
    )


def systems_to_anisoap_graph(
    systems: Sequence[Any],
    neighbor_list_options: Any,
    *,
    rotations_key: str = "rotations",
    ellipsoid_lengths_key: str = "ellipsoid_lengths",
) -> AniSOAPGraph:
    """Convert metatomic Systems into the tensor graph consumed by AniSOAP."""
    all_R, all_i, all_j = [], [], []
    all_species, all_structures, all_atoms, all_rot, all_len = [], [], [], [], []
    atom_offset = 0

    for system_i, system in enumerate(systems):
        species = system.types.to(dtype=torch.long)
        n_atoms = int(species.shape[0])
        R, i, j = _extract_neighbor_list_edges(
            system, neighbor_list_options, system_i, atom_offset
        )
        all_R.append(R)
        all_i.append(i)
        all_j.append(j)
        all_species.append(species)
        all_structures.append(
            torch.full((n_atoms,), system_i, device=species.device, dtype=torch.long)
        )
        all_atoms.append(torch.arange(n_atoms, device=species.device, dtype=torch.long))
        all_rot.append(
            _get_system_tensor_data(system, rotations_key).to(
                device=system.positions.device, dtype=system.positions.dtype
            )
        )
        all_len.append(
            _get_system_tensor_data(system, ellipsoid_lengths_key).to(
                device=system.positions.device, dtype=system.positions.dtype
            )
        )
        atom_offset += n_atoms

    if all_R:
        R_ij = torch.cat(all_R, dim=0)
        centers = torch.cat(all_i, dim=0)
        neighbors = torch.cat(all_j, dim=0)
    else:
        # Empty batch fallback
        device = all_species[0].device
        dtype = all_rot[0].dtype
        R_ij = torch.empty((0, 3), device=device, dtype=dtype)
        centers = torch.empty((0,), device=device, dtype=torch.long)
        neighbors = torch.empty((0,), device=device, dtype=torch.long)

    return AniSOAPGraph(
        R_ij=R_ij,
        centers=centers,
        neighbors=neighbors,
        species=torch.cat(all_species, dim=0),
        structures=torch.cat(all_structures, dim=0),
        atom_indices=torch.cat(all_atoms, dim=0),
        rotations=torch.cat(all_rot, dim=0),
        ellipsoid_lengths=torch.cat(all_len, dim=0),
    )


def frames_to_anisoap_graph(
    frames: Sequence[Any],
    *,
    cutoff_radius: float,
    rotation_key: str = "quaternion",
    rotation_type: str = "quaternion",
    subtract_center_contribution: bool = False,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> AniSOAPGraph:
    """Thin ASE adapter for tests and scripts.

    The main package boundary should be metatomic Systems or the direct tensor
    graph. This helper intentionally lives outside the differentiable path for
    quaternion conversion and ASE neighbor-list construction.
    """

    from ase.neighborlist import neighbor_list
    from scipy.spatial.transform import Rotation

    # Accept either a single ``ase.Atoms`` object or a sequence of frames.
    # Without this, iterating over a single Atoms object yields individual Atom
    # objects, which do not provide the frame-level APIs used below.
    if hasattr(frames, "get_atomic_numbers"):
        frames = [frames]

    all_R, all_i, all_j = [], [], []
    all_species, all_structures, all_atoms, all_rot, all_len = [], [], [], [], []
    atom_offset = 0
    for system_i, frame in enumerate(frames):
        numbers = torch.as_tensor(
            frame.get_atomic_numbers(), device=device, dtype=torch.long
        )
        positions = torch.as_tensor(frame.get_positions(), device=device, dtype=dtype)
        n_atoms = int(numbers.shape[0])

        i_np, j_np, S_np = neighbor_list(
            "ijS",
            frame,
            cutoff_radius,
            self_interaction=(not subtract_center_contribution),
        )
        if len(i_np):
            i_local = torch.as_tensor(i_np, device=device, dtype=torch.long)
            j_local = torch.as_tensor(j_np, device=device, dtype=torch.long)
            shifts = torch.as_tensor(S_np, device=device, dtype=dtype)
            cell = torch.as_tensor(frame.cell.array, device=device, dtype=dtype)
            R = positions[j_local] + shifts @ cell - positions[i_local]
            all_R.append(R)
            all_i.append(i_local + atom_offset)
            all_j.append(j_local + atom_offset)

        if rotation_type == "quaternion":
            key = rotation_key
            if (
                key not in frame.arrays
                and key == "c_q"
                and "quaternion" in frame.arrays
            ):
                key = "quaternion"

            if key not in frame.arrays:
                warnings.warn(
                    f"Frame {system_i} does not have rotations stored, using identity rotations."
                )
                rotations = torch.eye(3, device=device, dtype=dtype).repeat(
                    n_atoms, 1, 1
                )
            else:
                q = frame.arrays[key]
                matrices = np.asarray(
                    [Rotation.from_quat([*qq[1:], qq[0]]).as_matrix() for qq in q]
                )
                rotations = torch.as_tensor(matrices, device=device, dtype=dtype)
        elif rotation_type == "matrix":
            if rotation_key not in frame.arrays:
                warnings.warn(
                    f"Frame {system_i} does not have rotations stored, using identity rotations."
                )
                rotations = torch.eye(3, device=device, dtype=dtype).repeat(
                    n_atoms, 1, 1
                )
            else:
                rotations = torch.as_tensor(
                    frame.arrays[rotation_key], device=device, dtype=dtype
                )
        else:
            raise ValueError("rotation_type must be 'quaternion' or 'matrix'")

        lengths = torch.stack(
            [
                torch.as_tensor(
                    frame.arrays["c_diameter[1]"], device=device, dtype=dtype
                )
                / 2,
                torch.as_tensor(
                    frame.arrays["c_diameter[2]"], device=device, dtype=dtype
                )
                / 2,
                torch.as_tensor(
                    frame.arrays["c_diameter[3]"], device=device, dtype=dtype
                )
                / 2,
            ],
            dim=1,
        )
        all_species.append(numbers)
        all_structures.append(
            torch.full((n_atoms,), system_i, device=device, dtype=torch.long)
        )
        all_atoms.append(torch.arange(n_atoms, device=device, dtype=torch.long))
        all_rot.append(rotations)
        all_len.append(lengths)
        atom_offset += n_atoms

    if all_R:
        R_ij = torch.cat(all_R, dim=0)
        centers = torch.cat(all_i, dim=0)
        neighbors = torch.cat(all_j, dim=0)
    else:
        R_ij = torch.empty((0, 3), device=device, dtype=dtype)
        centers = torch.empty((0,), device=device, dtype=torch.long)
        neighbors = torch.empty((0,), device=device, dtype=torch.long)

    return AniSOAPGraph(
        R_ij=R_ij,
        centers=centers,
        neighbors=neighbors,
        species=torch.cat(all_species, dim=0),
        structures=torch.cat(all_structures, dim=0),
        atom_indices=torch.cat(all_atoms, dim=0),
        rotations=torch.cat(all_rot, dim=0),
        ellipsoid_lengths=torch.cat(all_len, dim=0),
    )


# -----------------------------------------------------------------------------
# Torch-native AniSOAP scientific pipeline
# -----------------------------------------------------------------------------


def pairwise_ellip_expansion(
    lmax: int,
    R_ij: torch.Tensor,
    centers: torch.Tensor,
    neighbors: torch.Tensor,
    species: torch.Tensor,
    structures: torch.Tensor,
    atom_indices: torch.Tensor,
    rotation_matrices: torch.Tensor,
    ellipsoid_lengths: torch.Tensor,
    sph_to_cart,
    radial_basis,
    types: List[int],
    num_ns: List[int],
    normalize: bool = True,
) -> TensorMap:
    r"""Torch-native pairwise expansion ``<a n l m | rho_ij>``.

    Returns the same scientific object as the original implementation, but with
    torch values and direct tensor-graph inputs.
    """
    R_ij = torch.as_tensor(R_ij)
    device, dtype = R_ij.device, R_ij.dtype
    centers = torch.as_tensor(centers, device=device, dtype=torch.long)
    neighbors = torch.as_tensor(neighbors, device=device, dtype=torch.long)
    species = torch.as_tensor(species, device=device, dtype=torch.long)
    structures = torch.as_tensor(structures, device=device, dtype=torch.long)
    atom_indices = torch.as_tensor(atom_indices, device=device, dtype=torch.long)
    rotation_matrices = torch.as_tensor(rotation_matrices, device=device, dtype=dtype)
    ellipsoid_lengths = torch.as_tensor(ellipsoid_lengths, device=device, dtype=dtype)

    if types is None:
        types = sorted(int(x) for x in torch.unique(species).detach().cpu().tolist())
    else:
        types = [int(x) for x in types]

    # num_ns = radial_basis.get_num_radial_functions()
    maxdeg = 0
    for l in range(lmax + 1):
        candidate = l + 2 * (int(num_ns[l]) - 1)
        if candidate > maxdeg:
            maxdeg = candidate
    scaled_sph_to_cart = []
    for l in range(lmax + 1):
        prefactor = math.sqrt((4.0 * math.pi) / (2 * l + 1))
        scaled_sph_to_cart.append(
            torch.as_tensor(sph_to_cart[l], device=device, dtype=dtype) / prefactor
        )

    # Accumulate values per (center_type, neighbor_type, l).
    values: Dict[Tuple[int, int, int], List[torch.Tensor]] = {}
    samples: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {}

    for edge in range(int(R_ij.shape[0])):
        i = int(centers[edge].detach().cpu().item())
        j = int(neighbors[edge].detach().cpu().item())
        center_type = int(species[i].detach().cpu().item())
        neighbor_type = int(species[j].detach().cpu().item())
        if center_type not in types or neighbor_type not in types:
            continue

        lengths = ellipsoid_lengths[j]
        rot = rotation_matrices[j]
        gaussian_norm = torch.as_tensor(
            (2.0 * math.pi) ** 1.5, device=device, dtype=dtype
        )
        length_norm = torch.reciprocal(torch.prod(lengths) * gaussian_norm)
        precision, center, constant = gaussian_parameters(
            radial_basis, R_ij[edge], lengths, rot
        )
        moments = compute_moments(precision, center, maxdeg)
        moments = moments * torch.exp(-0.5 * constant) * length_norm

        sample = (
            int(structures[i].detach().cpu().item()),
            int(atom_indices[i].detach().cpu().item()),
            int(atom_indices[j].detach().cpu().item()),
            int(edge),
        )
        for l in range(lmax + 1):
            n_l = num_ns[l]
            deg = l + 2 * (n_l - 1)
            moments_l = moments[: deg + 1, : deg + 1, : deg + 1]
            vals = torch.einsum("mnpqr,pqr->mn", scaled_sph_to_cart[l], moments_l)
            if normalize:
                vals = vals @ orthonormalization_matrix(
                    radial_basis, l, device=device, dtype=dtype
                )
            key = (center_type, neighbor_type, l)
            values.setdefault(key, []).append(vals)
            samples.setdefault(key, []).append(sample)

    keys: List[Tuple[int, int, int]] = []
    blocks: List[TensorBlock] = []
    for center_type in types:
        for neighbor_type in types:
            for l in range(lmax + 1):
                key = (center_type, neighbor_type, l)
                if key not in values:
                    continue
                block_values = torch.stack(values[key], dim=0)
                blocks.append(
                    TensorBlock(
                        values=block_values,
                        samples=Labels(
                            ["system", "first_atom", "second_atom", "pair"],
                            torch.as_tensor(
                                samples[key], device=device, dtype=torch.int32
                            ),
                        ),
                        components=[
                            Labels(
                                ["spherical_component_m"],
                                torch.arange(
                                    -l, l + 1, device=device, dtype=torch.int32
                                ).reshape(-1, 1),
                            )
                        ],
                        properties=Labels(
                            ["n"],
                            torch.arange(
                                num_ns[l], device=device, dtype=torch.int32
                            ).reshape(-1, 1),
                        ),
                    )
                )
                keys.append(key)

    if len(keys) == 0:
        key_values = torch.empty((0, 3), device=device, dtype=torch.int32)
    else:
        key_values = torch.as_tensor(keys, device=device, dtype=torch.int32)

    return TensorMap(
        keys=Labels(
            ["types_center", "types_neighbor", "angular_channel"],
            key_values,
        ),
        blocks=blocks,
    )


@torch.jit.ignore
def contract_pairwise_feat(
    pair_ellip_feat: TensorMap, types: Sequence[int]
) -> TensorMap:
    r"""Sum pairwise coefficients over neighbors of each type.

    This is the torch-native equivalent of the original
    ``contract_pairwise_feat`` and returns ``<a n l m | rho_i>`` blocks with
    keys ``(types_center, angular_channel)``.
    """
    types = [int(t) for t in types]
    key_names = pair_ellip_feat.keys.names
    pair_blocks: Dict[Tuple[int, int, int], TensorBlock] = {}
    for key, block in pair_ellip_feat.items():
        pair_blocks[
            (
                _key_value(key, key_names, "types_center"),
                _key_value(key, key_names, "types_neighbor"),
                _key_value(key, key_names, "angular_channel"),
            )
        ] = block

    out_keys: List[Tuple[int, int]] = []
    out_blocks: List[TensorBlock] = []
    center_l_keys = sorted({(tc, l) for (tc, _tn, l) in pair_blocks})

    for center_type, l in center_l_keys:
        grouped_by_neighbor: List[
            Tuple[int, List[Tuple[int, int]], torch.Tensor, torch.Tensor]
        ] = []
        all_sample_set = set()
        components = None
        device = dtype = None
        for neighbor_type in types:
            block = pair_blocks.get((center_type, neighbor_type, l))
            if block is None:
                continue
            sample_names = list(block.samples.names)
            sv = block.samples.values.to(dtype=torch.long, device=block.values.device)
            sys_col = sample_names.index("system")
            atom_col = sample_names.index("first_atom")
            sample_pairs = [
                (int(s), int(a))
                for s, a in sv[:, [sys_col, atom_col]].detach().cpu().tolist()
            ]
            unique_pairs = sorted(set(sample_pairs))
            pair_to_row = {p: r for r, p in enumerate(unique_pairs)}
            reduced = torch.zeros(
                (len(unique_pairs),) + tuple(block.values.shape[1:]),
                device=block.values.device,
                dtype=block.values.dtype,
            )
            rows = torch.tensor(
                [pair_to_row[p] for p in sample_pairs],
                device=block.values.device,
                dtype=torch.long,
            )
            reduced.index_add_(0, rows, block.values)
            grouped_by_neighbor.append(
                (neighbor_type, unique_pairs, reduced, block.properties.values)
            )
            all_sample_set.update(unique_pairs)
            components = block.components
            device = block.values.device
            dtype = block.values.dtype

        if not grouped_by_neighbor:
            continue
        all_samples = sorted(all_sample_set)
        all_index = {p: i for i, p in enumerate(all_samples)}
        value_chunks = []
        prop_chunks = []
        for neighbor_type, unique_pairs, reduced, props in grouped_by_neighbor:
            chunk = torch.zeros(
                (len(all_samples),) + tuple(reduced.shape[1:]),
                device=device,
                dtype=dtype,
            )
            rows = torch.tensor(
                [all_index[p] for p in unique_pairs], device=device, dtype=torch.long
            )
            chunk.index_copy_(0, rows, reduced)
            value_chunks.append(chunk)
            prop_chunks.append(
                torch.cat(
                    [
                        props.to(device=device, dtype=torch.int32),
                        torch.full(
                            (props.shape[0], 1),
                            neighbor_type,
                            device=device,
                            dtype=torch.int32,
                        ),
                    ],
                    dim=1,
                )
            )

        values = torch.cat(value_chunks, dim=2)
        properties = torch.cat(prop_chunks, dim=0)
        out_blocks.append(
            TensorBlock(
                values=values,
                samples=Labels(
                    ["type", "center"],
                    torch.as_tensor(all_samples, device=device, dtype=torch.int32),
                ),
                components=components,
                properties=Labels(
                    ["n", "neighbor_types"],
                    properties.to(device=device, dtype=torch.int32),
                ),
            )
        )
        out_keys.append((center_type, l))

    device = out_blocks[0].values.device if out_blocks else None
    return TensorMap(
        keys=Labels(
            ["types_center", "angular_channel"],
            torch.as_tensor(out_keys, device=device, dtype=torch.int32),
        ),
        blocks=out_blocks,
    )


class EllipsoidalDensityProjection(torch.nn.Module):
    """Torch-native AniSOAP ellipsoidal density projection.

    Main inputs are metatomic ``System`` objects or the explicit tensor graph used
    by AniSOAP-BPNN. The output can be either the scientific TensorMap power
    spectrum or a single-block per-atom feature TensorMap for neural networks.
    """

    def __init__(
        self,
        max_angular: int,
        radial_basis_name: str,
        cutoff_radius: float,
        *,
        compute_gradients: bool = False,
        subtract_center_contribution: bool = False,
        radial_gaussian_width: Optional[float] = None,
        max_radial: Optional[int | List[int]] = None,
        rotation_key: str = "quaternion",
        rotation_type: str = "quaternion",
        basis_rcond: float = 0.0,
        basis_tol: float = 1e-8,
        species: Optional[Sequence[int]] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if compute_gradients:
            raise NotImplementedError("Sorry! Gradients have not yet been implemented")
        if isinstance(cutoff_radius, int):
            raise ValueError(
                "r_cut is set as an integer, which could cause overflow errors. Pass in float"
            )
        if rotation_type not in ["quaternion", "matrix"]:
            raise ValueError("rotation_type must be 'quaternion' or 'matrix'")

        self.max_angular = int(max_angular)
        self.cutoff_radius = float(cutoff_radius)
        self.subtract_center_contribution = bool(subtract_center_contribution)
        self.radial_basis_name = radial_basis_name
        self.rotation_key = rotation_key
        self.rotation_type = rotation_type
        self.species = None if species is None else [int(s) for s in species]
        self.dtype = dtype
        self.shape: Optional[int] = None

        radial_hypers = {
            "radial_gaussian_width": radial_gaussian_width,
            "max_angular": self.max_angular,
            "cutoff_radius": self.cutoff_radius,
            "max_radial": max_radial,
            "rcond": basis_rcond,
            "tol": basis_tol,
        }
        if radial_basis_name == "gto":
            if radial_gaussian_width is None:
                raise ValueError("Gaussian width must be provided with GTO basis")
            if isinstance(radial_gaussian_width, int):
                raise ValueError(
                    "radial_gaussian_width is set as an integer, which could cause overflow errors. Pass in float."
                )
            self.radial_basis = GTORadialBasis(**radial_hypers)
        elif radial_basis_name == "monomial":
            if radial_gaussian_width is not None:
                raise ValueError("Gaussian width can only be provided with GTO basis")
            radial_hypers.pop("radial_gaussian_width")
            self.radial_basis = MonomialBasis(**radial_hypers)
        else:
            raise NotImplementedError("radial_basis_name must be 'gto' or 'monomial'")

        self.num_ns = self.radial_basis.get_num_radial_functions()
        self.sph_to_cart = spherical_to_cartesian(self.max_angular, self.num_ns)
        self._cg = TorchClebschGordanReal(self.max_angular)
        self._neighbor_list_options = None

    @torch.jit.ignore
    def requested_neighbor_lists(self) -> List[Any]:
        """Return the metatomic neighbor-list request for this descriptor."""
        try:
            from metatomic.torch import NeighborListOptions
        except Exception:
            from metatomic.torch.system import NeighborListOptions

        if self._neighbor_list_options is None:
            try:
                self._neighbor_list_options = NeighborListOptions(
                    cutoff=self.cutoff_radius,
                    full_list=True,
                    strict=False,
                )
            except TypeError:
                self._neighbor_list_options = NeighborListOptions(
                    cutoff=self.cutoff_radius,
                    full_list=True,
                )
        return [self._neighbor_list_options]

    def _graph_from_inputs(
        self,
        systems=None,
        frames=None,
        R_ij: Optional[torch.Tensor] = None,
        centers: Optional[torch.Tensor] = None,
        neighbors: Optional[torch.Tensor] = None,
        species: Optional[torch.Tensor] = None,
        structures: Optional[torch.Tensor] = None,
        atom_indices: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        ellipsoid_lengths: Optional[torch.Tensor] = None,
    ) -> AniSOAPGraph:
        if systems is not None:
            return systems_to_anisoap_graph(systems, self.requested_neighbor_lists()[0])
        if frames is not None:
            return frames_to_anisoap_graph(
                frames,
                cutoff_radius=self.cutoff_radius,
                rotation_key=self.rotation_key,
                rotation_type=self.rotation_type,
                subtract_center_contribution=self.subtract_center_contribution,
                dtype=self.dtype,
            )
        required = [R_ij, centers, neighbors, species, rotations, ellipsoid_lengths]
        if any(x is None for x in required):
            raise ValueError(
                "Provide systems=..., frames=..., or all tensor graph inputs: "
                "R_ij, centers, neighbors, species, rotations, ellipsoid_lengths"
            )
        R_ij = torch.as_tensor(R_ij)
        device = R_ij.device
        n_atoms = int(torch.as_tensor(species).shape[0])
        if structures is None:
            structures = torch.zeros((n_atoms,), device=device, dtype=torch.long)
        if atom_indices is None:
            atom_indices = torch.arange(n_atoms, device=device, dtype=torch.long)
        return AniSOAPGraph(
            R_ij=R_ij,
            centers=torch.as_tensor(centers, device=device, dtype=torch.long),
            neighbors=torch.as_tensor(neighbors, device=device, dtype=torch.long),
            species=torch.as_tensor(species, device=device, dtype=torch.long),
            structures=torch.as_tensor(structures, device=device, dtype=torch.long),
            atom_indices=torch.as_tensor(atom_indices, device=device, dtype=torch.long),
            rotations=torch.as_tensor(rotations, device=device, dtype=R_ij.dtype),
            ellipsoid_lengths=torch.as_tensor(
                ellipsoid_lengths, device=device, dtype=R_ij.dtype
            ),
        )

    @torch.jit.ignore
    def pairwise_expansion(
        self,
        frames=None,
        *,
        systems=None,
        R_ij: Optional[torch.Tensor] = None,
        centers: Optional[torch.Tensor] = None,
        neighbors: Optional[torch.Tensor] = None,
        species: Optional[torch.Tensor] = None,
        structures: Optional[torch.Tensor] = None,
        atom_indices: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        ellipsoid_lengths: Optional[torch.Tensor] = None,
        normalize: bool = True,
        show_progress: bool = False,
        **unused: Any,
    ) -> TensorMap:
        """Return the pairwise AniSOAP expansion coefficients.

        ``show_progress`` is accepted for API compatibility; the torch-native
        implementation currently does not use tqdm progress bars.
        """
        del show_progress, unused
        graph = self._graph_from_inputs(
            systems=systems,
            frames=frames,
            R_ij=R_ij,
            centers=centers,
            neighbors=neighbors,
            species=species,
            structures=structures,
            atom_indices=atom_indices,
            rotations=rotations,
            ellipsoid_lengths=ellipsoid_lengths,
        )
        types = self.species or sorted(
            int(x) for x in torch.unique(graph.species).detach().cpu().tolist()
        )
        return pairwise_ellip_expansion(
            self.max_angular,
            graph.R_ij,
            graph.centers,
            graph.neighbors,
            graph.species,
            graph.structures,
            graph.atom_indices,
            graph.rotations,
            graph.ellipsoid_lengths,
            self.sph_to_cart,
            self.radial_basis,
            types=types,
            normalize=normalize,
            num_ns=self.radial_basis.get_num_radial_functions(),
        )

    def transform(
        self,
        frames=None,
        *,
        systems=None,
        R_ij: Optional[torch.Tensor] = None,
        centers: Optional[torch.Tensor] = None,
        neighbors: Optional[torch.Tensor] = None,
        species: Optional[torch.Tensor] = None,
        structures: Optional[torch.Tensor] = None,
        atom_indices: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        ellipsoid_lengths: Optional[torch.Tensor] = None,
        normalize: bool = True,
        show_progress: bool = False,
        return_pairwise: bool = False,
        return_pef: Optional[bool] = None,
        **unused: Any,
    ) -> TensorMap | Tuple[TensorMap, TensorMap]:
        """Return the contracted nu=1 AniSOAP expansion coefficients."""
        del show_progress, unused
        if return_pef is not None:
            return_pairwise = return_pef
        graph = self._graph_from_inputs(
            systems=systems,
            frames=frames,
            R_ij=R_ij,
            centers=centers,
            neighbors=neighbors,
            species=species,
            structures=structures,
            atom_indices=atom_indices,
            rotations=rotations,
            ellipsoid_lengths=ellipsoid_lengths,
        )
        types = self.species or sorted(
            int(x) for x in torch.unique(graph.species).detach().cpu().tolist()
        )
        pairwise = pairwise_ellip_expansion(
            self.max_angular,
            graph.R_ij,
            graph.centers,
            graph.neighbors,
            graph.species,
            graph.structures,
            graph.atom_indices,
            graph.rotations,
            graph.ellipsoid_lengths,
            self.sph_to_cart,
            self.radial_basis,
            types=types,
            normalize=normalize,
            num_ns=self.radial_basis.get_num_radial_functions(),
        )
        coeffs = contract_pairwise_feat(pairwise, types)
        if return_pairwise:
            return coeffs, pairwise
        return coeffs

    @torch.jit.ignore
    def power_spectrum(
        self,
        frames=None,
        *,
        systems=None,
        mean_over_samples: bool = True,
        show_progress: bool = False,
        normalize: bool = True,
        **kwargs: Any,
    ) -> TensorMap | torch.Tensor:
        """Return the scientific AniSOAP nu=2 power spectrum.

        With ``mean_over_samples=False`` this returns the full TensorMap. With
        ``mean_over_samples=True`` this returns a dense system-level feature
        tensor, matching the old convenience behavior but without NumPy.
        """
        coeffs = self.transform(
            frames,
            systems=systems,
            normalize=normalize,
            show_progress=show_progress,
            **kwargs,
        )
        nu1 = standardize_keys(coeffs)
        nu2 = cg_combine(
            nu1,
            nu1,
            clebsch_gordan=self._cg,
            lcut=0,
            other_keys_match=["types_center"],
        )
        if not mean_over_samples:
            return nu2
        features, samples = self.power_spectrum_features_from_tensormap(
            nu2, aggregate_by_system=True
        )
        return features

    def power_spectrum_features_from_tensormap(
        self,
        nu2: TensorMap,
        *,
        target_samples: Optional[Labels] = None,
        aggregate_by_system: bool = False,
    ) -> Tuple[torch.Tensor, Labels]:
        """Flatten a nu=2 TensorMap into a dense feature matrix.

        This is the BPNN-facing representation. Columns from blocks that do not
        apply to a sample are filled with zeros, preserving center-species blocks
        without losing differentiability.
        """
        # Determine the row index set.
        sample_set = set()
        if target_samples is not None:
            sample_tuples = _labels_to_tuples(target_samples)
        else:
            for _, block in nu2.items():
                sample_set.update(_labels_to_tuples(block.samples))
            sample_tuples = sorted(sample_set)
        sample_index = {s: i for i, s in enumerate(sample_tuples)}

        # Determine total feature dimension and allocate.
        block_dims = []
        total = 0
        device = nu2.block(0).values.device
        dtype = nu2.block(0).values.dtype
        for _, block in nu2.items():
            dim = int(block.values.shape[1] * block.values.shape[2])
            block_dims.append(dim)
            total += dim
        dense = torch.zeros((len(sample_tuples), total), device=device, dtype=dtype)

        col = 0
        for (_, block), dim in zip(nu2.items(), block_dims):
            vals = block.values.reshape(block.values.shape[0], dim)
            rows = torch.tensor(
                [sample_index[s] for s in _labels_to_tuples(block.samples)],
                device=device,
                dtype=torch.long,
            )

            block_dense = dense[:, col : col + dim]
            block_dense.index_copy_(0, rows, vals)
            dense = torch.cat(
                [dense[:, :col], block_dense, dense[:, col + dim :]],
                dim=1,
            )

            col += dim

        samples = (
            target_samples
            if target_samples is not None
            else Labels(
                list(nu2.block(0).samples.names),
                torch.as_tensor(sample_tuples, device=device, dtype=torch.int32),
            )
        )
        if aggregate_by_system:
            names = list(samples.names)
            if "system" in names:
                system_col = names.index("system")
            elif "type" in names:
                system_col = names.index("type")
            else:
                raise ValueError(
                    "aggregate_by_system=True requires a 'system' or 'type' sample label"
                )

            systems = samples.values[:, system_col].to(device=device, dtype=torch.long)
            n_systems = (
                int(systems.max().detach().cpu().item()) + 1 if systems.numel() else 0
            )
            out = torch.zeros((n_systems, dense.shape[1]), device=device, dtype=dtype)
            counts = torch.zeros((n_systems, 1), device=device, dtype=dtype)
            out.index_add_(0, systems, dense)
            counts.index_add_(
                0,
                systems,
                torch.ones((systems.shape[0], 1), device=device, dtype=dtype),
            )
            return out / counts.clamp_min(1), Labels(
                ["system"],
                torch.arange(n_systems, device=device, dtype=torch.int32).reshape(
                    -1, 1
                ),
            )
        return dense, samples

    @torch.jit.ignore
    def power_spectrum_features(
        self, aggregate_by_system: bool = False, **kwargs: Any
    ) -> Tuple[torch.Tensor, Labels]:
        """Return dense AniSOAP power-spectrum features and their sample labels."""
        nu2 = self.power_spectrum(mean_over_samples=False, **kwargs)
        return self.power_spectrum_features_from_tensormap(
            nu2, aggregate_by_system=aggregate_by_system
        )

    def _feature_size(self, *, device=None, dtype=None) -> int:
        """Infer the dense power-spectrum feature dimension from one dummy edge."""
        if self.shape is not None:
            return int(self.shape)

        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = torch.device("cpu")

        center_type = self.species[0] if self.species is not None else 0
        dummy_features = self.power_spectrum_feature_tensor_map(
            torch.zeros((1, 3), device=device, dtype=dtype),
            torch.tensor([0], device=device, dtype=torch.long),
            torch.tensor([0], device=device, dtype=torch.long),
            torch.tensor([center_type], device=device, dtype=torch.long),
            torch.tensor([0], device=device, dtype=torch.long),
            torch.tensor([0], device=device, dtype=torch.long),
            torch.eye(3, device=device, dtype=dtype).reshape(1, 3, 3),
            torch.ones((1, 3), device=device, dtype=dtype),
            True,
        )
        self.shape = int(dummy_features.block(0).values.shape[1])
        return self.shape

    @torch.jit.export
    def power_spectrum_feature_tensor_map(
        self,
        R_ij: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        species: torch.Tensor,
        structures: torch.Tensor,
        atom_indices: torch.Tensor,
        rotations: torch.Tensor,
        ellipsoid_lengths: torch.Tensor,
        normalize: bool = True,
    ) -> TensorMap:
        """Return a single-block per-atom feature TensorMap for AniSOAP-BPNN.

        The block layout is ``samples=['system', 'atom']`` and
        ``properties=['property']``, matching the SOAP-BPNN scalar descriptor
        interface.
        """
        R_ij = torch.as_tensor(R_ij)
        device = R_ij.device
        dtype = R_ij.dtype

        centers = torch.as_tensor(centers, device=device, dtype=torch.long)
        neighbors = torch.as_tensor(neighbors, device=device, dtype=torch.long)
        species = torch.as_tensor(species, device=device, dtype=torch.long)
        structures = torch.as_tensor(structures, device=device, dtype=torch.long)
        atom_indices = torch.as_tensor(atom_indices, device=device, dtype=torch.long)
        rotations = torch.as_tensor(rotations, device=device, dtype=dtype)
        ellipsoid_lengths = torch.as_tensor(
            ellipsoid_lengths,
            device=device,
            dtype=dtype,
        )

        target_samples = Labels(
            ["system", "atom"],
            torch.stack(
                [
                    structures.to(dtype=torch.int32),
                    atom_indices.to(dtype=torch.int32),
                ],
                dim=1,
            ),
        )

        if R_ij.shape[0] == 0:
            if self.shape is None:
                self.shape = self._feature_size()

            all_species = (
                self.species
                if self.species is not None
                else sorted(
                    int(x) for x in torch.unique(species).detach().cpu().tolist()
                )
            )

            blocks = torch.jit.annotate(List[TensorBlock], [])
            keys = torch.jit.annotate(List[Tuple[int]], [])

            for center_type in all_species:
                mask = species == int(center_type)
                if not bool(mask.any()):
                    continue

                sample_values = torch.stack(
                    [
                        structures[mask].to(device=R_ij.device, dtype=torch.int32),
                        atom_indices[mask].to(device=R_ij.device, dtype=torch.int32),
                    ],
                    dim=1,
                )

                blocks.append(
                    TensorBlock(
                        values=torch.zeros(
                            (sample_values.shape[0], self.shape),
                            device=R_ij.device,
                            dtype=R_ij.dtype,
                        ),
                        samples=Labels(["system", "atom"], sample_values),
                        components=[],
                        properties=Labels(
                            ["property"],
                            torch.arange(
                                self.shape,
                                device=R_ij.device,
                                dtype=torch.int32,
                            ).reshape(-1, 1),
                        ),
                    )
                )
                keys.append((int(center_type),))

            return TensorMap(
                keys=Labels(
                    ["center_type"],
                    torch.as_tensor(keys, device=R_ij.device, dtype=torch.int32),
                ),
                blocks=blocks,
            )

        if self.species is None:
            raise RuntimeError(
                "TorchScript path requires EllipsoidalDensityProjection.species to be set."
            )
        types = self.species

        pairwise = pairwise_ellip_expansion(
            self.max_angular,
            R_ij,
            centers,
            neighbors,
            species,
            structures,
            atom_indices,
            rotations,
            ellipsoid_lengths,
            self.sph_to_cart,
            self.radial_basis,
            types=types,
            normalize=normalize,
            num_ns=self.radial_basis.get_num_radial_functions(),
        )

        coeffs = contract_pairwise_feat(pairwise, types)
        nu1 = standardize_keys(coeffs)
        nu2 = cg_combine(
            nu1,
            nu1,
            clebsch_gordan=self._cg,
            lcut=0,
            other_keys_match=["types_center"],
        )
        features, _ = self.power_spectrum_features_from_tensormap(
            nu2, target_samples=target_samples
        )

        self.shape = int(features.shape[1])

        blocks = torch.jit.annotate(List[TensorBlock], [])
        keys = torch.jit.annotate(List[Tuple[int]], [])

        all_species = (
            self.species
            if self.species is not None
            else sorted(int(x) for x in torch.unique(species).detach().cpu().tolist())
        )

        for center_type in all_species:
            mask = species == int(center_type)
            if not bool(mask.any()):
                continue

            sample_values = torch.stack(
                [
                    structures[mask].to(device=features.device, dtype=torch.int32),
                    atom_indices[mask].to(device=features.device, dtype=torch.int32),
                ],
                dim=1,
            )

            target_rows = {
                tuple(int(v) for v in row.detach().cpu().tolist()): idx
                for idx, row in enumerate(target_samples.values)
            }
            row_indices = torch.tensor(
                [
                    target_rows[tuple(int(v) for v in row.detach().cpu().tolist())]
                    for row in sample_values
                ],
                device=features.device,
                dtype=torch.long,
            )

            blocks.append(
                TensorBlock(
                    values=features.index_select(0, row_indices),
                    samples=Labels(["system", "atom"], sample_values),
                    components=[],
                    properties=Labels(
                        ["property"],
                        torch.arange(
                            features.shape[1],
                            device=features.device,
                            dtype=torch.int32,
                        ).reshape(-1, 1),
                    ),
                )
            )
            keys.append((int(center_type),))

        return TensorMap(
            keys=Labels(
                ["center_type"],
                torch.as_tensor(keys, device=features.device, dtype=torch.int32),
            ),
            blocks=blocks,
        )

    def forward(
        self,
        R_ij: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        species: torch.Tensor,
        structures: torch.Tensor,
        atom_indices: torch.Tensor,
        rotations: torch.Tensor,
        ellipsoid_lengths: torch.Tensor,
        normalize: bool = True,
    ) -> TensorMap:
        """Default module output for AniSOAP-BPNN: per-atom scalar feature map."""
        return self.power_spectrum_feature_tensor_map(
            R_ij=R_ij,
            centers=centers,
            neighbors=neighbors,
            species=species,
            structures=structures,
            atom_indices=atom_indices,
            rotations=rotations,
            ellipsoid_lengths=ellipsoid_lengths,
            normalize=normalize,
        )


__all__ = [
    "AniSOAPGraph",
    "EllipsoidalDensityProjection",
    "TorchClebschGordanReal",
    "pairwise_ellip_expansion",
    "contract_pairwise_feat",
    "standardize_keys",
    "cg_combine",
    "systems_to_anisoap_graph",
    "frames_to_anisoap_graph",
]
