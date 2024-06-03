import re

import numpy as np
import wigners
from metatensor import (
    Labels,
    TensorBlock,
    TensorMap,
)

from .cyclic_list import CGRCacheList


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
    """
    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics(coefficients) of order L.

    It's meant to be applied to the left, ``real2complex @ [-L..L]``.
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


def _remove_suffix(names, new_suffix=""):
    suffix = re.compile("_[0-9]?$")
    rname = []
    for name in names:
        match = suffix.search(name)
        if match is None:
            rname.append(name + new_suffix)
        else:
            rname.append(name[: match.start()] + new_suffix)
    return rname


def standardize_keys(descriptor):
    """Standardize the naming scheme of density expansion coefficient blocks (nu=1)"""

    key_names = descriptor.keys.names
    if not "angular_channel" in key_names:
        raise ValueError(
            "Descriptor missing spherical harmonics channel key `angular_channel`"
        )
    blocks = []
    keys = []
    for key, block in descriptor.items():
        key = tuple(key)
        if not "order_nu" in key_names:
            key = (1,) + key
        keys.append(key)
        property_names = _remove_suffix(block.properties.names, "_1")
        blocks.append(
            TensorBlock(
                values=block.values,
                samples=block.samples,
                components=block.components,
                properties=Labels(
                    property_names,
                    np.asarray(block.properties, dtype=np.int32).reshape(
                        -1, len(property_names)
                    ),
                ),
            )
        )

    if not "order_nu" in key_names:
        key_names = ["order_nu"] + key_names

    return TensorMap(
        keys=Labels(names=key_names, values=np.asarray(keys, dtype=np.int32)),
        blocks=blocks,
    )


def cg_combine(
    x_a,
    x_b,
    feature_names=None,
    clebsch_gordan=None,
    lcut=None,
    other_keys_match=None,
):
    """
    Performs a CG product of two sets of equivariants. Only requirement is that
    sparse indices are labeled as ("inversion_sigma", "spherical_harmonics_l", "order_nu"). The automatically-determined
    naming of output features can be overridden by giving a list of "feature_names".
    By defaults, all other key labels are combined in an "outer product" mode, i.e. if there is a key-side
    neighbor_types in both x_a and x_b, the returned keys will have two neighbor_types labels,
    corresponding to the parent features. By providing a list `other_keys_match` of keys that should match, these are
    not outer-producted, but combined together. for instance, passing `["types center"]` means that the keys with the
    same types center will be combined together, but yield a single key with the same types_center in the results.
    """

    # determines the cutoff in the new features
    lmax_a = np.asarray(x_a.keys["angular_channel"], dtype="int32").max()
    lmax_b = np.asarray(x_b.keys["angular_channel"], dtype="int32").max()
    if lcut is None:
        lcut = lmax_a + lmax_b

    # creates a CG object, if needed
    if clebsch_gordan is None:
        clebsch_gordan = ClebschGordanReal(lcut)

    other_keys_a = tuple(
        name for name in x_a.keys.names if name not in ["angular_channel", "order_nu"]
    )
    other_keys_b = tuple(
        name for name in x_b.keys.names if name not in ["angular_channel", "order_nu"]
    )

    if other_keys_match is None:
        OTHER_KEYS = [k + "_a" for k in other_keys_a] + [k + "_b" for k in other_keys_b]
    else:
        OTHER_KEYS = (
            other_keys_match
            + [
                k + ("_a" if k in other_keys_b else "")
                for k in other_keys_a
                if k not in other_keys_match
            ]
            + [
                k + ("_b" if k in other_keys_a else "")
                for k in other_keys_b
                if k not in other_keys_match
            ]
        )

    # we assume grad components are all the same
    if x_a.block(0).has_gradient("positions"):
        grad_components = x_a.block(0).gradient("positions").components
    else:
        grad_components = None

    # automatic generation of the output features names
    # "x1 x2 x3 ; x1 x2 -> x1_a x2_a x3_a k_nu x1_b x2_b l_nu"
    if feature_names is None:
        NU = x_a.keys[0]["order_nu"] + x_b.keys[0]["order_nu"]
        feature_names = (
            tuple(n + "_a" for n in x_a.block(0).properties.names)
            + ("k_" + str(NU),)
            + tuple(n + "_b" for n in x_b.block(0).properties.names)
            + ("l_" + str(NU),)
        )

    X_idx = {}
    X_blocks = {}
    X_samples = {}
    X_grad_samples = {}
    X_grads = {}

    # loops over sparse blocks of x_a
    for index_a, block_a in x_a.items():
        lam_a = index_a["angular_channel"]
        order_a = index_a["order_nu"]
        properties_a = (
            block_a.properties
        )  # pre-extract this block as accessing a c property has a non-zero cost
        samples_a = block_a.samples

        # and x_b
        for index_b, block_b in x_b.items():
            lam_b = index_b["angular_channel"]
            order_b = index_b["order_nu"]
            properties_b = block_b.properties
            samples_b = block_b.samples

            if other_keys_match is None:
                OTHERS = tuple(index_a[name] for name in other_keys_a) + tuple(
                    index_b[name] for name in other_keys_b
                )
            else:
                OTHERS = tuple(
                    index_a[k] for k in other_keys_match if index_a[k] == index_b[k]
                )
                # skip combinations without matching key
                if len(OTHERS) < len(other_keys_match):
                    continue
                # adds non-matching keys to build outer product
                OTHERS = OTHERS + tuple(
                    index_a[k] for k in other_keys_a if k not in other_keys_match
                )
                OTHERS = OTHERS + tuple(
                    index_b[k] for k in other_keys_b if k not in other_keys_match
                )

            neighbor_slice = slice(None)

            # determines the properties that are in the select list
            sel_feats = []
            sel_idx = []
            sel_feats = (
                np.indices((len(properties_a), len(properties_b))).reshape(2, -1).T
            )

            prop_ids_a = []
            prop_ids_b = []
            for n_a, f_a in enumerate(properties_a):
                prop_ids_a.append(tuple(f_a) + (lam_a,))
            for n_b, f_b in enumerate(properties_b):
                prop_ids_b.append(tuple(f_b) + (lam_b,))
            prop_ids_a = np.asarray(prop_ids_a)
            prop_ids_b = np.asarray(prop_ids_b)
            sel_idx = np.hstack(
                [prop_ids_a[sel_feats[:, 0]], prop_ids_b[sel_feats[:, 1]]]
            )
            if len(sel_feats) == 0:
                continue
            # loops over all permissible output blocks. note that blocks will
            # be filled from different la, lb
            for L in range(np.abs(lam_a - lam_b), 1 + min(lam_a + lam_b, lcut)):
                # determines parity of the block
                NU = order_a + order_b
                KEY = (
                    NU,
                    L,
                ) + OTHERS
                if not KEY in X_idx:
                    X_idx[KEY] = []
                    X_blocks[KEY] = []
                    X_samples[KEY] = block_b.samples
                    if grad_components is not None:
                        X_grads[KEY] = []
                        X_grad_samples[KEY] = block_b.gradient("positions").samples

                # builds all products in one go
                one_shot_blocks = clebsch_gordan.combine_einsum(
                    block_a.values[neighbor_slice][:, :, sel_feats[:, 0]],
                    block_b.values[:, :, sel_feats[:, 1]],
                    L,
                    combination_string="iq,iq->iq",
                )
                # do gradients, if they are present...
                if grad_components is not None:
                    grad_a = block_a.gradient("positions")
                    grad_b = block_b.gradient("positions")
                    grad_a_data = np.swapaxes(grad_a.data, 1, 2)
                    grad_b_data = np.swapaxes(grad_b.data, 1, 2)
                    one_shot_grads = clebsch_gordan.combine_einsum(
                        block_a.values[grad_a.samples["sample"]][
                            neighbor_slice, :, sel_feats[:, 0]
                        ],
                        grad_b_data[..., sel_feats[:, 1]],
                        L=L,
                        combination_string="iq,iaq->iaq",
                    ) + clebsch_gordan.combine_einsum(
                        block_b.values[grad_b.samples["sample"]][:, :, sel_feats[:, 1]],
                        grad_a_data[neighbor_slice, ..., sel_feats[:, 0]],
                        L=L,
                        combination_string="iq,iaq->iaq",
                    )

                # now loop over the selected features to build the blocks

                X_idx[KEY].append(sel_idx)
                X_blocks[KEY].append(one_shot_blocks)
                if grad_components is not None:
                    X_grads[KEY].append(one_shot_grads)

    # turns data into sparse storage format (and dumps any empty block in the process)
    nz_idx = []
    nz_blk = []
    for KEY in X_blocks:
        L = KEY[1]
        # create blocks
        if len(X_blocks[KEY]) == 0:
            continue  # skips empty blocks
        nz_idx.append(KEY)
        block_data = np.concatenate(X_blocks[KEY], axis=-1)
        sph_components = Labels(
            ["spherical_harmonics_m"],
            np.asarray(range(-L, L + 1), dtype=np.int32).reshape(-1, 1),
        )
        newblock = TensorBlock(
            # feature index must be last
            values=block_data,
            samples=X_samples[KEY],
            components=[sph_components],
            properties=Labels(
                feature_names, np.asarray(np.vstack(X_idx[KEY]), dtype=np.int32)
            ),
        )
        if grad_components is not None:
            grad_data = np.swapaxes(np.concatenate(X_grads[KEY], axis=-1), 2, 1)
            newblock.add_gradient(
                "positions",
                data=grad_data,
                samples=X_grad_samples[KEY],
                components=[grad_components[0], sph_components],
            )
        nz_blk.append(newblock)
    X = TensorMap(
        Labels(
            ["order_nu", "angular_channel"] + OTHER_KEYS,
            np.asarray(nz_idx, dtype=np.int32),
        ),
        nz_blk,
    )
    return X
