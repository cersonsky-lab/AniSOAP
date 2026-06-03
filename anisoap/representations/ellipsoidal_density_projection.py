"""Torch-native ellipsoidal density projection for AniSOAP."""

from __future__ import annotations

import math
import re
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
import wigners
from metatensor.torch import (
    Labels,
    TensorBlock,
    TensorMap,
)

from anisoap.representations.radial_basis import (
    GTORadialBasis,
    MonomialBasis,
)
from anisoap.utils.spherical_to_cartesian import spherical_to_cartesian

from .radial_basis import (
    gaussian_parameters,
    orthonormalization_matrix,
)


def _row_tuple(row: Any) -> Tuple[int, ...]:
    values = getattr(row, "values", None)
    if values is not None and not callable(values):
        row = values

    if torch.is_tensor(row):
        return tuple(int(v) for v in row.detach().cpu().reshape(-1).tolist())

    if hasattr(row, "__iter__") and not isinstance(row, (str, bytes)):
        return tuple(int(v) for v in row)

    return (int(row),)


def _labels_to_tuples(labels: Labels) -> List[Tuple[int, ...]]:
    values = labels.values.detach().cpu()
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return [tuple(int(v) for v in row.tolist()) for row in values]


def _key_tuple(key: Any) -> Tuple[int, ...]:
    if torch.is_tensor(key):
        return _row_tuple(key)

    values = getattr(key, "values", None)
    if values is not None and not callable(values):
        return _row_tuple(values)

    return _row_tuple(key)


def _key_value(key: Any, key_names: Sequence[str], name: str) -> int:
    try:
        return int(key[name])
    except Exception:
        idx = list(key_names).index(name)
        return _key_tuple(key)[idx]


def _remove_suffix(names: Sequence[str], new_suffix: str = "") -> List[str]:
    suffix = re.compile(r"_[0-9]?$")
    out = []
    for name in names:
        match = suffix.search(name)
        out.append(
            name + new_suffix if match is None else name[: match.start()] + new_suffix
        )
    return out


def compute_moments(A: torch.Tensor, a: torch.Tensor, maxdeg: int) -> torch.Tensor:
    r"""Differentiable trivariate Gaussian moments.

    Computes moments of ``exp(-1/2 (x-a)^T A (x-a))`` up to total polynomial
    degree ``maxdeg``. This replaces the original Rust ``compute_moments`` call
    on the torch path.
    """
    A = torch.as_tensor(A)
    a = torch.as_tensor(a, device=A.device, dtype=A.dtype)
    if A.shape != (3, 3):
        raise ValueError(f"A must have shape (3, 3), got {tuple(A.shape)}")
    if a.shape != (3,):
        raise ValueError(f"a must have shape (3,), got {tuple(a.shape)}")
    if maxdeg < 0:
        raise ValueError("maxdeg must be non-negative")

    device, dtype = A.device, A.dtype
    cov = torch.linalg.inv(A)
    norm = torch.as_tensor(
        (2.0 * math.pi) ** 1.5, device=device, dtype=dtype
    ) / torch.sqrt(torch.linalg.det(A))

    # Store normalized raw moments first; multiply by the Gaussian integral at end.
    M: Dict[Tuple[int, int, int], torch.Tensor] = {
        (0, 0, 0): torch.ones((), device=device, dtype=dtype)
    }
    if maxdeg >= 1:
        M[(1, 0, 0)] = a[0]
        M[(0, 1, 0)] = a[1]
        M[(0, 0, 1)] = a[2]
    if maxdeg >= 2:
        M[(2, 0, 0)] = cov[0, 0] + a[0] * a[0]
        M[(0, 2, 0)] = cov[1, 1] + a[1] * a[1]
        M[(0, 0, 2)] = cov[2, 2] + a[2] * a[2]
        M[(1, 1, 0)] = cov[0, 1] + a[0] * a[1]
        M[(0, 1, 1)] = cov[1, 2] + a[1] * a[2]
        M[(1, 0, 1)] = cov[0, 2] + a[0] * a[2]

    def get(i: int, j: int, k: int) -> torch.Tensor:
        if i < 0 or j < 0 or k < 0 or i + j + k > maxdeg:
            return torch.zeros((), device=device, dtype=dtype)
        return M.get((i, j, k), torch.zeros((), device=device, dtype=dtype))

    # Isserlis/Stein recurrence: E[X_p f(X)] = mu_p E[f] + sum_q Sigma_pq E[df/dx_q]
    for degree in range(2, maxdeg):
        updates: Dict[Tuple[int, int, int], torch.Tensor] = {}
        for n0 in range(degree + 1):
            for n1 in range(degree + 1 - n0):
                n2 = degree - n0 - n1
                base = get(n0, n1, n2)
                updates[(n0 + 1, n1, n2)] = (
                    a[0] * base
                    + cov[0, 0] * n0 * get(n0 - 1, n1, n2)
                    + cov[0, 1] * n1 * get(n0, n1 - 1, n2)
                    + cov[0, 2] * n2 * get(n0, n1, n2 - 1)
                )
                if n0 == 0:
                    updates[(n0, n1 + 1, n2)] = (
                        a[1] * base
                        + cov[1, 0] * n0 * get(n0 - 1, n1, n2)
                        + cov[1, 1] * n1 * get(n0, n1 - 1, n2)
                        + cov[1, 2] * n2 * get(n0, n1, n2 - 1)
                    )
                    if n1 == 0:
                        updates[(n0, n1, n2 + 1)] = (
                            a[2] * base
                            + cov[2, 0] * n0 * get(n0 - 1, n1, n2)
                            + cov[2, 1] * n1 * get(n0, n1 - 1, n2)
                            + cov[2, 2] * n2 * get(n0, n1, n2 - 1)
                        )
        M.update(updates)

    out = torch.zeros((maxdeg + 1, maxdeg + 1, maxdeg + 1), device=device, dtype=dtype)
    if M:
        idx = torch.tensor(list(M.keys()), device=device, dtype=torch.long).T
        vals = torch.stack(list(M.values())) * norm
        out = out.index_put(tuple(idx), vals, accumulate=False)
    return out


def _real2complex(L: int) -> np.ndarray:
    result = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.complex128)
    inv_sqrt_2 = 1.0 / np.sqrt(2.0)
    for m in range(-L, L + 1):
        if m < 0:
            result[L - m, L + m] = inv_sqrt_2 * 1j * (-1) ** m
            result[L + m, L + m] = -inv_sqrt_2 * 1j
        elif m == 0:
            result[L, L] = 1.0
        else:
            result[L + m, L + m] = inv_sqrt_2 * (-1) ** m
            result[L - m, L + m] = inv_sqrt_2
    return result


def _complex_clebsch_gordan_matrix(l1: int, l2: int, L: int) -> np.ndarray:
    if abs(l1 - l2) > L or abs(l1 + l2) < L:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    return wigners.clebsch_gordan_array(l1, l2, L)


class TorchClebschGordanReal:
    """Real Clebsch-Gordan coefficients with torch-valued contractions."""

    def __init__(self, l_max: int):
        self.l_max = int(l_max)
        self._cg: Dict[
            Tuple[int, int, int], List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        ] = {}
        self._init_cg()

    def _init_cg(self) -> None:
        r2c = {L: _real2complex(L) for L in range(self.l_max + 1)}
        c2r = {L: np.conjugate(r2c[L]).T for L in range(self.l_max + 1)}
        for l1 in range(self.l_max + 1):
            for l2 in range(self.l_max + 1):
                for L in range(abs(l1 - l2), min(self.l_max, l1 + l2) + 1):
                    complex_cg = _complex_clebsch_gordan_matrix(l1, l2, L)
                    real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
                        complex_cg.shape
                    )
                    real_cg = real_cg.swapaxes(0, 1)
                    real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(
                        real_cg.shape
                    )
                    real_cg = real_cg.swapaxes(0, 1)
                    real_cg = real_cg @ c2r[L].T
                    rcg = (
                        np.real(real_cg) if (l1 + l2 + L) % 2 == 0 else np.imag(real_cg)
                    )
                    sparse_by_M = []
                    for M in range(2 * L + 1):
                        nz = np.where(np.abs(rcg[:, :, M]) > 1e-15)
                        sparse_by_M.append(
                            (
                                nz[0].astype(np.int64),
                                nz[1].astype(np.int64),
                                rcg[nz[0], nz[1], M].astype(np.float64),
                            )
                        )
                    self._cg[(l1, l2, L)] = sparse_by_M

    def combine_einsum(
        self,
        rho1: torch.Tensor,
        rho2: torch.Tensor,
        L: int,
        combination_string: str = "iq,iq->iq",
    ) -> torch.Tensor:
        l1 = (rho1.shape[1] - 1) // 2
        l2 = (rho2.shape[1] - 1) // 2
        if (l1, l2, L) not in self._cg:
            raise ValueError(f"Requested CG entry {(l1, l2, L)} was not precomputed")
        if rho1.shape[0] != rho2.shape[0]:
            raise ValueError("Cannot combine blocks with different number of samples")

        # The AniSOAP power spectrum path uses only this product form. Keeping the
        # argument makes the implementation line up with the original utility.
        if combination_string != "iq,iq->iq":
            raise NotImplementedError(
                "Only 'iq,iq->iq' is implemented on the torch path"
            )

        n_samples = rho1.shape[0]
        n_features = rho1.shape[-1]
        out = torch.zeros(
            (n_samples, 2 * L + 1, n_features), device=rho1.device, dtype=rho1.dtype
        )
        for M, (m1s, m2s, cgs) in enumerate(self._cg[(l1, l2, L)]):
            if len(cgs) == 0:
                continue
            val = torch.zeros(
                (n_samples, n_features), device=rho1.device, dtype=rho1.dtype
            )
            for m1, m2, cg in zip(m1s, m2s, cgs):
                val = val + rho1[:, int(m1), :] * rho2[:, int(m2), :] * torch.as_tensor(
                    cg, device=rho1.device, dtype=rho1.dtype
                )
            out[:, M, :] = val
        return out


def standardize_keys(descriptor: TensorMap) -> TensorMap:
    """Torch-native version of AniSOAP's ``standardize_keys``."""
    key_names = descriptor.keys.names
    if "angular_channel" not in key_names:
        raise ValueError("Descriptor missing key 'angular_channel'")

    blocks: List[TensorBlock] = []
    keys: List[Tuple[int, ...]] = []
    for key, block in descriptor.items():
        key_t = _key_tuple(key)
        if "order_nu" not in key_names:
            key_t = (1,) + key_t
        keys.append(key_t)
        prop_names = _remove_suffix(block.properties.names, "_1")
        blocks.append(
            TensorBlock(
                values=block.values,
                samples=block.samples,
                components=block.components,
                properties=Labels(
                    prop_names, block.properties.values.to(dtype=torch.int32)
                ),
            )
        )

    if "order_nu" not in key_names:
        key_names = ["order_nu"] + key_names
    device = blocks[0].values.device if blocks else None
    return TensorMap(
        keys=Labels(
            key_names,
            torch.as_tensor(keys, device=device, dtype=torch.int32),
        ),
        blocks=blocks,
    )


def cg_combine(
    x_a: TensorMap,
    x_b: TensorMap,
    *,
    feature_names: Optional[Sequence[str]] = None,
    clebsch_gordan: Optional[TorchClebschGordanReal] = None,
    lcut: Optional[int] = None,
    other_keys_match: Optional[Sequence[str]] = None,
) -> TensorMap:
    """Torch-native port of AniSOAP's metatensor CG product.

    This keeps the original key/property semantics while replacing NumPy products
    with torch operations so gradients flow through coefficient values.
    """
    key_names_a = x_a.keys.names
    key_names_b = x_b.keys.names
    lmax_a = int(x_a.keys.values[:, key_names_a.index("angular_channel")].max().item())
    lmax_b = int(x_b.keys.values[:, key_names_b.index("angular_channel")].max().item())
    if lcut is None:
        lcut = lmax_a + lmax_b
    if clebsch_gordan is None:
        clebsch_gordan = TorchClebschGordanReal(lcut)

    other_a = tuple(n for n in key_names_a if n not in ["angular_channel", "order_nu"])
    other_b = tuple(n for n in key_names_b if n not in ["angular_channel", "order_nu"])
    if other_keys_match is None:
        output_other_keys = [k + "_a" for k in other_a] + [k + "_b" for k in other_b]
    else:
        output_other_keys = (
            list(other_keys_match)
            + [
                k + ("_a" if k in other_b else "")
                for k in other_a
                if k not in other_keys_match
            ]
            + [
                k + ("_b" if k in other_a else "")
                for k in other_b
                if k not in other_keys_match
            ]
        )

    if feature_names is None:
        first_key = next(iter(x_a.keys))
        second_key = next(iter(x_b.keys))
        order_nu = _key_value(first_key, key_names_a, "order_nu") + _key_value(
            second_key, key_names_b, "order_nu"
        )
        feature_names = (
            tuple(n + "_a" for n in x_a.block(0).properties.names)
            + ("k_" + str(order_nu),)
            + tuple(n + "_b" for n in x_b.block(0).properties.names)
            + ("l_" + str(order_nu),)
        )

    X_idx: Dict[Tuple[int, ...], List[torch.Tensor]] = {}
    X_blocks: Dict[Tuple[int, ...], List[torch.Tensor]] = {}
    X_samples: Dict[Tuple[int, ...], Labels] = {}

    for index_a, block_a in x_a.items():
        lam_a = _key_value(index_a, key_names_a, "angular_channel")
        order_a = _key_value(index_a, key_names_a, "order_nu")
        props_a = block_a.properties.values

        for index_b, block_b in x_b.items():
            lam_b = _key_value(index_b, key_names_b, "angular_channel")
            order_b = _key_value(index_b, key_names_b, "order_nu")
            props_b = block_b.properties.values

            if other_keys_match is None:
                others = tuple(
                    _key_value(index_a, key_names_a, k) for k in other_a
                ) + tuple(_key_value(index_b, key_names_b, k) for k in other_b)
            else:
                matched = []
                skip = False
                for k in other_keys_match:
                    va = _key_value(index_a, key_names_a, k)
                    vb = _key_value(index_b, key_names_b, k)
                    if va != vb:
                        skip = True
                        break
                    matched.append(va)
                if skip:
                    continue
                others = tuple(matched)
                others += tuple(
                    _key_value(index_a, key_names_a, k)
                    for k in other_a
                    if k not in other_keys_match
                )
                others += tuple(
                    _key_value(index_b, key_names_b, k)
                    for k in other_b
                    if k not in other_keys_match
                )

            # Original code assumes matching samples. Enforce this explicitly.
            if (
                block_a.samples.values.shape != block_b.samples.values.shape
                or not torch.equal(
                    block_a.samples.values.to(block_b.samples.values.device),
                    block_b.samples.values,
                )
            ):
                raise ValueError(
                    "CG combination requires matching samples in the two blocks"
                )

            n_pa = props_a.shape[0]
            n_pb = props_b.shape[0]
            if n_pa == 0 or n_pb == 0:
                continue
            grid = torch.cartesian_prod(
                torch.arange(n_pa, device=block_a.values.device, dtype=torch.long),
                torch.arange(n_pb, device=block_a.values.device, dtype=torch.long),
            )
            prop_ids_a = torch.cat(
                [
                    props_a.to(block_a.values.device, dtype=torch.int32),
                    torch.full(
                        (n_pa, 1),
                        lam_a,
                        device=block_a.values.device,
                        dtype=torch.int32,
                    ),
                ],
                dim=1,
            )
            prop_ids_b = torch.cat(
                [
                    props_b.to(block_a.values.device, dtype=torch.int32),
                    torch.full(
                        (n_pb, 1),
                        lam_b,
                        device=block_a.values.device,
                        dtype=torch.int32,
                    ),
                ],
                dim=1,
            )
            sel_idx = torch.cat([prop_ids_a[grid[:, 0]], prop_ids_b[grid[:, 1]]], dim=1)

            vals_a = block_a.values[:, :, grid[:, 0]]
            vals_b = block_b.values[:, :, grid[:, 1]]
            for L in range(abs(lam_a - lam_b), min(lam_a + lam_b, lcut) + 1):
                key = (order_a + order_b, L) + tuple(int(v) for v in others)
                if key not in X_blocks:
                    X_idx[key] = []
                    X_blocks[key] = []
                    X_samples[key] = block_b.samples
                X_idx[key].append(sel_idx)
                X_blocks[key].append(
                    clebsch_gordan.combine_einsum(vals_a, vals_b, L, "iq,iq->iq")
                )

    keys_out: List[Tuple[int, ...]] = []
    blocks_out: List[TensorBlock] = []
    for key in sorted(X_blocks):
        if not X_blocks[key]:
            continue
        L = key[1]
        values = torch.cat(X_blocks[key], dim=-1)
        props = torch.cat(X_idx[key], dim=0).to(dtype=torch.int32)
        keys_out.append(key)
        blocks_out.append(
            TensorBlock(
                values=values,
                samples=X_samples[key],
                components=[
                    Labels(
                        ["spherical_harmonics_m"],
                        torch.arange(
                            -L, L + 1, dtype=torch.int32, device=values.device
                        ).reshape(-1, 1),
                    )
                ],
                properties=Labels(list(feature_names), props),
            )
        )

    device = blocks_out[0].values.device if blocks_out else None
    return TensorMap(
        keys=Labels(
            ["order_nu", "angular_channel"] + output_other_keys,
            torch.as_tensor(keys_out, device=device, dtype=torch.int32),
        ),
        blocks=blocks_out,
    )


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
    sph_to_cart: Sequence[np.ndarray],
    radial_basis: Any,
    *,
    types: Optional[Sequence[int]] = None,
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

    num_ns = radial_basis.get_num_radial_functions()
    maxdeg = int(np.max(np.arange(lmax + 1) + 2 * np.array(num_ns)))
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
        precision, center, constant = compute_gaussian_parameters(
            radial_basis, R_ij[edge], lengths, rot
        )
        moments = compute_moments(precision, center, maxdeg)
        moments = moments * torch.exp(-0.5 * constant) * length_norm

        sample = (
            int(structures[i].detach().cpu().item()),
            int(atom_indices[i].detach().cpu().item()),
            int(atom_indices[j].detach().cpu().item()),
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
                            ["system", "first_atom", "second_atom"],
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

    return TensorMap(
        keys=Labels(
            ["types_center", "types_neighbor", "angular_channel"],
            torch.as_tensor(keys, device=device, dtype=torch.int32),
        ),
        blocks=blocks,
    )


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

    def requested_neighbor_lists(self) -> List[Any]:
        """Return the metatomic neighbor-list request for this descriptor."""
        try:
            from metatomic.torch import NeighborListOptions
        except Exception:
            try:
                from metatomic.torch.system import NeighborListOptions
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    "metatomic.torch is required for requested_neighbor_lists"
                ) from exc
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
        *,
        systems: Optional[Sequence[Any]] = None,
        frames: Optional[Sequence[Any]] = None,
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

    def pairwise_expansion(
        self,
        frames: Optional[Sequence[Any]] = None,
        *,
        systems: Optional[Sequence[Any]] = None,
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
        )

    def transform(
        self,
        frames: Optional[Sequence[Any]] = None,
        *,
        systems: Optional[Sequence[Any]] = None,
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
        )
        coeffs = contract_pairwise_feat(pairwise, types)
        if return_pairwise:
            return coeffs, pairwise
        return coeffs

    def power_spectrum(
        self,
        frames: Optional[Sequence[Any]] = None,
        *,
        systems: Optional[Sequence[Any]] = None,
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
            dense.index_copy_(0, rows, vals)
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
            system_col = names.index("system")
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

    def power_spectrum_features(
        self, aggregate_by_system: bool = False, **kwargs: Any
    ) -> Tuple[torch.Tensor, Labels]:
        """Return dense AniSOAP power-spectrum features and their sample labels."""
        nu2 = self.power_spectrum(mean_over_samples=False, **kwargs)
        return self.power_spectrum_features_from_tensormap(
            nu2, aggregate_by_system=aggregate_by_system
        )

    def power_spectrum_feature_tensor_map(self, **kwargs: Any) -> TensorMap:
        """Return a single-block per-atom feature TensorMap for AniSOAP-BPNN.

        The block layout is ``samples=['system', 'atom']`` and
        ``properties=['property']``, matching the SOAP-BPNN scalar descriptor
        interface.
        """
        graph = self._graph_from_inputs(**kwargs)
        target_samples = Labels(
            ["system", "atom"],
            torch.stack(
                [graph.structures.to(torch.int32), graph.atom_indices.to(torch.int32)],
                dim=1,
            ).to(device=graph.R_ij.device, dtype=torch.int32),
        )
        # Reuse graph tensors to avoid reconstructing systems/frames.
        nu2 = self.power_spectrum(
            mean_over_samples=False,
            R_ij=graph.R_ij,
            centers=graph.centers,
            neighbors=graph.neighbors,
            species=graph.species,
            structures=graph.structures,
            atom_indices=graph.atom_indices,
            rotations=graph.rotations,
            ellipsoid_lengths=graph.ellipsoid_lengths,
        )
        features, _ = self.power_spectrum_features_from_tensormap(
            nu2, target_samples=target_samples
        )

        self.shape = int(features.shape[1])
        return TensorMap(
            keys=Labels(
                ["_"], torch.tensor([[0]], device=features.device, dtype=torch.int32)
            ),
            blocks=[
                TensorBlock(
                    values=features,
                    samples=target_samples,
                    components=[],
                    properties=Labels(
                        ["property"],
                        torch.arange(
                            features.shape[1], device=features.device, dtype=torch.int32
                        ).reshape(-1, 1),
                    ),
                )
            ],
        )

    def forward(self, **kwargs: Any) -> TensorMap:
        """Default module output for AniSOAP-BPNN: per-atom scalar feature map."""
        return self.power_spectrum_feature_tensor_map(**kwargs)


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
