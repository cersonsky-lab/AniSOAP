import re
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

from .cyclic_list import CGRCacheList


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


class ClebschGordanReal:
    # Size of 5 is an arbitrary choice -- it can be changed to any number.
    # Set to None to disable caching.
    cache_list = CGRCacheList(5)

    def __init__(self, l_max):
        self._l_max = l_max
        self._cg = dict()

        # Check if the caching feature is activated.
        if ClebschGordanReal.cache_list is not None:
            # Check if the given l_max is already in the cache.
            if self._l_max in ClebschGordanReal.cache_list.keys():
                # If so, load from the cache.
                self._cg = ClebschGordanReal.cache_list.get_val(self._l_max)
            else:
                # Otherwise, compute the matrices and store it to the cache.
                self._init_cg()
                ClebschGordanReal.cache_list.insert(self._l_max, self._cg)
        else:
            # If caching is deactivated, then just compute the matrices normally.
            self._init_cg()

    def _init_cg(self):
        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        for L in range(0, self._l_max + 1):
            r2c[L] = _real2complex(L)
            c2r[L] = np.conjugate(r2c[L]).T

        for l1 in range(self._l_max + 1):
            for l2 in range(self._l_max + 1):
                for L in range(
                    max(l1, l2) - min(l1, l2), min(self._l_max, (l1 + l2)) + 1
                ):
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

                    if (l1 + l2 + L) % 2 == 0:
                        rcg = np.real(real_cg)
                    else:
                        rcg = np.imag(real_cg)

                    new_cg = []
                    for M in range(2 * L + 1):
                        cg_nonzero = np.where(np.abs(rcg[:, :, M]) > 1e-15)
                        cg_M = np.zeros(
                            len(cg_nonzero[0]),
                            dtype=[("m1", ">i4"), ("m2", ">i4"), ("cg", ">f8")],
                        )
                        cg_M["m1"] = cg_nonzero[0]
                        cg_M["m2"] = cg_nonzero[1]
                        cg_M["cg"] = rcg[cg_nonzero[0], cg_nonzero[1], M]
                        new_cg.append(cg_M)

                    self._cg[(l1, l2, L)] = new_cg

    def get_cg(self):
        return self._cg

    def combine_einsum(self, rho1, rho2, L, combination_string):
        # automatically infer l1 and l2 from the size of the coefficients vectors
        l1 = (rho1.shape[1] - 1) // 2
        l2 = (rho2.shape[1] - 1) // 2
        if L > self._l_max or l1 > self._l_max or l2 > self._l_max:
            print(self._l_max, L, l1, l2)
            raise ValueError(
                "Requested CG entry ", (l1, l2, L), " has not been precomputed"
            )

        n_items = rho1.shape[0]
        if rho1.shape[0] != rho2.shape[0]:
            raise IndexError(
                "Cannot combine feature blocks with different number of items"
            )

        # infers the shape of the output using the einsum internals
        features = np.einsum(combination_string, rho1[:, 0, ...], rho2[:, 0, ...]).shape
        rho = np.zeros((n_items, 2 * L + 1) + features[1:])

        if (l1, l2, L) in self._cg:
            for M in range(2 * L + 1):
                for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                    rho[:, M, ...] += np.einsum(
                        combination_string,
                        rho1[:, m1, ...],
                        rho2[:, m2, ...] * cg,
                    )

        return rho


def _real2complex(L):
    r"""Computes a matrix that converts from real to complex coefficients.

    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics (coefficients) of order `L`.

    Parameters
    ----------
    L : int
        The order of the spherical harmonics for which the matrix will be computed.

    Returns
    -------
    np.ndarray
        Matrix that can be used to convert from real to complex-valued spherical
        harmonics of order `L`.

    Note
    ----
    The matrix generated is meant to be applied from the left; that is, if :math:`\mathbf{R}`
    is the matrix returned from the function, it should be used like this:

    .. math::

        \mathbf{R} \dot \left[-L..L\right]

    """
    result = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.complex128)

    I_SQRT_2 = 1.0 / np.sqrt(2)

    for m in range(-L, L + 1):
        if m < 0:
            result[L - m, L + m] = I_SQRT_2 * 1j * (-1) ** m
            result[L + m, L + m] = -I_SQRT_2 * 1j

        if m == 0:
            result[L, L] = 1.0

        if m > 0:
            result[L + m, L + m] = I_SQRT_2 * (-1) ** m
            result[L - m, L + m] = I_SQRT_2

    return result


def _complex_clebsch_gordan_matrix(l1, l2, L):
    if np.abs(l1 - l2) > L or np.abs(l1 + l2) < L:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    else:
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
