import sys
import warnings
from itertools import product

import metatensor
import numpy as np
from anisoap_rust_lib import compute_moments
from metatensor import (
    Labels,
    TensorBlock,
    TensorMap,
)
from rascaline import NeighborList
from scipy.spatial.transform import Rotation
from skmatter.preprocessing import StandardFlexibleScaler
from tqdm.auto import tqdm

from anisoap.representations.radial_basis import (
    GTORadialBasis,
    MonomialBasis,
)
from anisoap.utils.metatensor_utils import (
    ClebschGordanReal,
    cg_combine,
    standardize_keys,
)
from anisoap.utils.moment_generator import *
from anisoap.utils.spherical_to_cartesian import spherical_to_cartesian


def pairwise_ellip_expansion(
    lmax,
    neighbor_list,
    types,
    frame_to_global_atom_idx,
    rotation_matrices,
    ellipsoid_lengths,
    sph_to_cart,
    radial_basis,
    show_progress=False,
    rust_moments=True,
):
    r"""Computes pairwise expansion

    Function to compute the pairwise expansion :math:`\langle anlm|\rho_{ij} \rangle`
    by combining the moments and the spherical to Cartesian transformation.

    Parameters
    ----------
    lmax : int
        Maximum angular
    neighbor_list : metatensor.TensorMap
        Full neighborlist with keys (types_1, types_2) enumerating the possible
        species pairs. Each block contains as samples, the atom indices of
        (first_atom, second_atom) that correspond to the key, and block.value is
        a 3D array of the form (num_samples, num_components,properties), with
        num_components=3 corresponding to the (x,y,z) components of the vector
        from first_atom to second_atom. Depending on the cutoff some species
        pairs may not appear. Self pairs are not present but in PBC, pairs between
        copies of the same atom are accounted for.
    types : list of ints
        List of atomic numbers present across the data frames
    frame_to_global_atom_idx : list of ints
        The length of the list equals the number of frames, each entry enumerating
        the number of atoms in corresponding frame
    rotation_matrices : np.array of dimension ((num_atoms,3,3))
    ellipsoid_lengths : np.array of dimension ((num_atoms,3))
    radial_basis : Instance of the RadialBasis Class
        anisoap.representations.radial_basis.RadialBasis that has been instantiated
        appropriately with the cutoff radius, radial basis type.
    show_progress : bool
        Show progress bar for frame analysis and feature generation
    rust_moments : bool
        Use the ported rust code, which should result in increased speed. Default = True.
        In the future, once we ensure integrity checks with the original python code,
        this kwarg will be deprecated, and the rust version will always be used.

    Returns
    -------
    TensorMap
        A metatensor TensorMap with keys (types_1, types_2, l) where
        ("types_1", "types_2") is key in the neighbor_list and "l" is the
        angular channel. Each block of this tensormap has the same samples as the
        corresponding block of the neighbor_list. block.value is a 3D array of
        the form (num_samples, num_components, properties) where num_components
        form the 2*l+1 values for the corresponding angular channel and the
        properties dimension corresponds to the radial channel.
    """
    tensorblock_list = []
    keys = np.asarray(neighbor_list.keys, dtype=int)
    keys = [tuple(i) + (l,) for i in keys for l in range(lmax + 1)]
    num_ns = radial_basis.get_num_radial_functions()
    maxdeg = np.max(np.arange(lmax + 1) + 2 * np.array(num_ns))

    # This prefactor is the solid harmonics prefactor, that we need to divide by later.
    # This is needed because spherical_to_cartesian calculates solid harmonics Rlm = sqrt((4pi)/(2l+1)) * r^l*Ylm
    # Our expansion coefficients from the inner product does not have this prefactor included, so we divide it later.
    solid_harm_prefact = np.sqrt((4 * np.pi) / (np.arange(lmax + 1) * 2 + 1))
    scaled_sph_to_cart = []
    for l in range(lmax + 1):
        scaled_sph_to_cart.append(sph_to_cart[l] / solid_harm_prefact[l])

    for center_types in types:
        for neighbor_types in types:
            if (center_types, neighbor_types) in neighbor_list.keys:
                values_ldict = {l: [] for l in range(lmax + 1)}
                nl_block = neighbor_list.block(
                    first_atom_type=center_types,
                    second_atom_type=neighbor_types,
                )

                for isample, nl_sample in enumerate(
                    tqdm(
                        nl_block.samples,
                        disable=(not show_progress),
                        desc="Iterating samples for ({}, {})".format(
                            center_types, neighbor_types
                        ),
                        leave=False,
                    )
                ):
                    frame_idx, i, j = (
                        nl_sample["system"],
                        nl_sample["first_atom"],
                        nl_sample["second_atom"],
                    )
                    i_global = frame_to_global_atom_idx[frame_idx] + i
                    j_global = frame_to_global_atom_idx[frame_idx] + j

                    r_ij = np.asarray(
                        [
                            nl_block.values[isample, 0],
                            nl_block.values[isample, 1],
                            nl_block.values[isample, 2],
                        ]
                    ).reshape(
                        3,
                    )

                    rot = rotation_matrices[j_global]
                    lengths = ellipsoid_lengths[j_global]
                    length_norm = (
                        np.prod(lengths) * (2.0 * np.pi) ** (3.0 / 2.0)
                    ) ** -1.0

                    (
                        precision,
                        center,
                        constant,
                    ) = radial_basis.compute_gaussian_parameters(r_ij, lengths, rot)

                    if rust_moments:
                        moments = compute_moments(precision, center, maxdeg)
                    else:
                        moments = compute_moments_inefficient_implementation(
                            precision, center, maxdeg=maxdeg
                        )
                    moments *= np.exp(-0.5 * constant) * length_norm

                    for l in range(lmax + 1):
                        deg = l + 2 * (num_ns[l] - 1)
                        moments_l = moments[: deg + 1, : deg + 1, : deg + 1]
                        values_ldict[l].append(
                            np.einsum(
                                "mnpqr, pqr->mn",
                                scaled_sph_to_cart[l],
                                moments_l,
                            )
                        )

                for l in tqdm(
                    range(lmax + 1),
                    disable=(not show_progress),
                    desc="Accruing lmax",
                    leave=False,
                ):
                    block = TensorBlock(
                        values=np.asarray(values_ldict[l]),
                        samples=nl_block.samples,  # as many rows as samples
                        components=[
                            Labels(
                                ["spherical_component_m"],
                                np.asarray([list(range(-l, l + 1))], np.int32).reshape(
                                    -1, 1
                                ),
                            )
                        ],
                        properties=Labels(
                            ["n"],
                            np.asarray(list(range(num_ns[l])), np.int32).reshape(-1, 1),
                        ),
                    )
                    tensorblock_list.append(block)

    pairwise_ellip_feat = TensorMap(
        Labels(
            ["types_center", "types_neighbor", "angular_channel"],
            np.asarray(keys, dtype=np.int32),
        ),
        tensorblock_list,
    )
    return pairwise_ellip_feat


def contract_pairwise_feat(pair_ellip_feat, types, show_progress=False):
    """Function to sum over the pairwise expansion

    .. math::

        \\sum_{j \\in a} \\langle anlm|\\rho_{ij} \\rangle
            = \\langle anlm|\\rho_i \\rangle

    Parameters
    ----------
    pair_ellip_feat : metatensor.TensorMap
        TensorMap returned from "pairwise_ellip_expansion()" with keys
        (types_1, types_2,l) enumerating the possible species pairs and the
        angular channels.
    types: list of ints
        List of atomic numbers present across the data frames
    show_progress : bool
        Show progress bar for frame analysis and feature generation

    Returns
    -------
    TensorMap
        A metatensor TensorMap with keys (types, l) "types" takes the value
        of the atomic numbers present in the dataset and "l" is the angular
        channel. Each block of this tensormap has as samples ("type", "center"),
        yielding the indices of the frames and atoms that correspond to "species"
        are present. block.value is a 3D array of the form (num_samples, num_components, properties)
        where num_components take on the same values as in the pair_ellip_feat_feat.block.
        block.properties now has an additional index for neighbor_species that
        corresponds to "a" in :math:`\\langle anlm|rho_i \\rangle`
    """
    ellip_keys = list(
        set([tuple(list(x)[:1] + list(x)[2:]) for x in pair_ellip_feat.keys])
    )
    # Select the unique combinations of pair_ellip_feat.keys["types_center"] and
    # pair_ellip_feat.keys["angular_channel"] to form the keys of the single particle centered feature
    ellip_keys.sort()
    ellip_blocks = []
    property_names = pair_ellip_feat.property_names + [
        "neighbor_types",
    ]

    for key in tqdm(
        ellip_keys, disable=(not show_progress), desc="Iterating tensor block keys"
    ):
        contract_blocks = []
        contract_properties = []
        contract_samples = []
        # these collect the values, properties and samples of the blocks when contracted over neighbor_types.
        # All these lists have as many entries as len(types).

        for ele in tqdm(
            types,
            disable=(not show_progress),
            desc="Iterating neighbor types",
            leave=False,
        ):
            selection = Labels(names=["types_neighbor"], values=np.array([[ele]]))
            blockidx = pair_ellip_feat.blocks_matching(selection=selection)
            # indices of the blocks in pair_ellip_feat with neighbor types = ele
            sel_blocks = [
                pair_ellip_feat.block(i)
                for i in blockidx
                if key
                == tuple(
                    list(pair_ellip_feat.keys[i])[:1]
                    + list(pair_ellip_feat.keys[i])[2:]
                )
            ]

            if not len(sel_blocks):
                #                 print(key, ele, "skipped") # this block is not found in the pairwise feat
                continue
            assert len(sel_blocks) == 1

            # sel_blocks is the corresponding block in the pairwise feat with the same (types_center, l) and
            # types_neighbor = ele thus there can be only one block corresponding to the triplet (types_center, types_neighbor, l)
            block = sel_blocks[0]

            pair_block_sample = list(
                zip(block.samples["system"], block.samples["first_atom"])
            )

            # Takes the system and first atom index from the current pair_block sample. There might be repeated
            # entries here because for example (0,0,1) (0,0,2) might be samples of the pair block (the index of the
            # neighbor atom is changing but for both of these we are keeping (0,0) corresponding to the system and
            # first atom.

            struct, center = np.unique(block.samples["system"]), np.unique(
                block.samples["first_atom"]
            )
            possible_block_samples = list(product(struct, center))
            # possible block samples contains all *unique* possible pairwise products between system and atom index
            # From here we choose the entries that are actually present in the block to form the final sample

            block_samples = []
            block_values = []

            indexed_sample_idx = {}
            for idx, tup in enumerate(pair_block_sample):
                if tup not in indexed_sample_idx:
                    l = []
                else:
                    l = indexed_sample_idx[tup]
                l.append(idx)
                indexed_sample_idx[tup] = l

            for isample, sample in enumerate(
                tqdm(
                    possible_block_samples,
                    disable=(not show_progress),
                    desc="Finding matching block samples",
                    leave=False,
                )
            ):
                if sample in indexed_sample_idx:
                    sample_idx = indexed_sample_idx[tuple(sample)]

                    # all samples of the pair block that match the current sample
                    # in the example above, for sample = (0,0) we would identify sample_idx = [(0,0,1), (0,0,2)]
                    if len(sample_idx) == 0:
                        continue
                    #             #print(key, ele, sample, block.samples[sample_idx])
                    block_samples.append(sample)
                    block_values.append(
                        block.values[sample_idx].sum(axis=0)
                    )  # sum over "j"  for given ele

                    # block_values has as many entries as samples satisfying (key, neighbor_types=ele).
                    # When we iterate over neighbor types, not all (type, center) would be present
                    # Example: (0,0,1) might be present in a block with neighbor_types = 1 but no other pair block
                    # ever has (0,0,x) present as a sample- so (0,0) doesnt show up in a block_sample for all ele
                    # so in general we have a ragged list of contract_blocks

            contract_blocks.append(block_values)
            contract_samples.append(block_samples)
            contract_properties.append([tuple(p) + (ele,) for p in block.properties])
            # this adds the "ele" (i.e. neighbor_types) to the properties dimension

        #         print(len(contract_samples))
        all_block_samples = sorted(list(set().union(*contract_samples)))
        # Selects the set of samples from all the block_samples we collected by iterating over the neighbor_types
        # These form the final samples of the block!

        all_block_values = np.zeros(
            (
                (len(all_block_samples),)
                + block.values.shape[1:]
                + (len(contract_blocks),)
            )
        )
        # Create storage for the final values - we need as many rows as all_block_samples,
        # block.values.shape[1:] accounts for "components" and "properties" that are already part of the pair blocks
        # and we dont alter these
        # len(contract_blocks) - adds the additional dimension for the neighbor_types since we accumulated
        # values for each of them as \sum_{j in ele} <|rho_ij>
        #  Thus - all_block_values.shape = (num_final_samples, components_pair, properties_pair, num_types)

        indexed_elem_cont_samples = {}
        for i, val in enumerate(all_block_samples):
            if val not in indexed_elem_cont_samples:
                l = []
            else:
                l = indexed_elem_cont_samples[val]
            l.append(i)
            indexed_elem_cont_samples[val] = l

        for iele, elem_cont_samples in enumerate(
            tqdm(
                contract_samples,
                disable=(not show_progress),
                desc="Contracting features",
                leave=False,
            )
        ):
            # This effectively loops over the types of the neighbors
            # Now we just need to add the contributions to the final samples and values from this types to the right
            # samples
            nzidx = list(
                sorted(
                    [
                        i
                        for v in elem_cont_samples
                        for i in indexed_elem_cont_samples[tuple(v)]
                    ]
                )
            )

            # identifies where the samples that this types contributes to, are present in the final samples
            #             print(apecies[ib],key, bb, all_block_samples)
            all_block_values[nzidx, :, :, iele] = contract_blocks[iele]

        new_block = TensorBlock(
            values=all_block_values.reshape(
                all_block_values.shape[0], all_block_values.shape[1], -1
            ),
            samples=Labels(["type", "center"], np.asarray(all_block_samples, np.int32)),
            components=block.components,
            properties=Labels(
                list(property_names),
                np.asarray(np.vstack(contract_properties), np.int32),
            ),
        )

        ellip_blocks.append(new_block)
    ellip = TensorMap(
        Labels(
            ["types_center", "angular_channel"],
            np.asarray(ellip_keys, dtype=np.int32),
        ),
        ellip_blocks,
    )

    return ellip


class EllipsoidalDensityProjection:
    """Class for computing spherical projection coefficients.

    Compute the spherical projection coefficients for a system of ellipsoids
    assuming a multivariate Gaussian density.

    Initialize the calculator using the hyperparameters.

    Parameters
    ----------
    max_angular : int
        Number of angular functions
    radial_basis : _RadialBasis
        The radial basis. Currently implemented are
        'gto', 'monomial'.
    compute_gradients : bool
        Compute gradients
    subtract_center_contribution : bool
        Subtract contribution from the central atom.
    rotation_key : string
        Key under which rotations are stored in ase frames arrays
    rotation_type : string
        Type of rotation object being passed. Currently implemented
        are 'quaternion' and 'matrix'
    max_radial : None, int, list of int
        Number of radial bases to use. Can either correspond to number of
        bases per spherical harmonic or a value to use with every harmonic.
        If `None`, then for every `l`, `(max_angular - l) // 2 + 1` will
        be used.

    Attributes
    ----------
    features : numpy.ndarray
    feature_gradients : numpy.ndarray

    """

    def __init__(
        self,
        max_angular,
        radial_basis_name,
        cutoff_radius,
        compute_gradients=False,
        subtract_center_contribution=False,
        radial_gaussian_width=None,
        max_radial=None,
        rotation_key="quaternion",
        rotation_type="quaternion",
        basis_rcond=0,
        basis_tol=1e-8,
    ):
        """Instantiates an object of type EllipsoidalDensityProjection.

        Parameters
        ----------
        max_angular : int
            Number of angular functions
        radial_basis_name : str
            The radial basis. Currently implemented are 'GTO_primitive', 'GTO',
            and 'monomial'.
        cutoff_radius
            Cutoff radius of the projection
        compute_gradients : bool, optional
            Compute gradients; defaults to 'False'
        subtract_center_contribution : bool, optional
            Subtract contribution from the central atom.  Defaults to 'False'
        radial_gaussian_width : float, optional
            Width of the Gaussian
        max_radial : int, list of int
            Number of radial bases to use. Can either correspond to number of
            bases per spherical harmonic or a value to use with every harmonic.
            If `None`, then for every `l`, `(max_angular - l) // 2 + 1` will
            be used.
        rotation_key : string
            Key under which rotations are stored in ase frames arrays
        rotation_type : string
            Type of rotation object being passed. Currently implemented
            are 'quaternion' and 'matrix'

        """
        # Store the input variables
        self.max_angular = max_angular
        self.cutoff_radius = cutoff_radius
        self.compute_gradients = compute_gradients
        self.subtract_center_contribution = subtract_center_contribution
        self.radial_basis_name = radial_basis_name

        # Currently, gradients are not supported
        if compute_gradients:
            raise NotImplementedError("Sorry! Gradients have not yet been implemented")
        #

        radial_hypers = {}
        radial_hypers["radial_gaussian_width"] = radial_gaussian_width
        radial_hypers["max_angular"] = max_angular
        radial_hypers["cutoff_radius"] = cutoff_radius
        radial_hypers["max_radial"] = max_radial
        radial_hypers["rcond"] = basis_rcond
        radial_hypers["tol"] = basis_tol

        # Initialize the radial basis class
        if type(cutoff_radius) == int:
            raise ValueError(
                "r_cut is set as an integer, which could cause overflow errors. Pass in float"
            )
        if radial_basis_name == "gto":
            if radial_hypers.get("radial_gaussian_width") is None:
                raise ValueError("Gaussian width must be provided with GTO basis")
            if type(radial_hypers.get("radial_gaussian_width")) == int:
                raise ValueError(
                    "radial_gaussian_width is set as an integer, which could cause overflow errors. Pass in float."
                )
            self.radial_basis = GTORadialBasis(**radial_hypers)
        elif radial_basis_name == "monomial":
            rgw = radial_hypers.pop("radial_gaussian_width")
            if rgw is not None:
                raise ValueError("Gaussian width can only be provided with GTO basis")
            self.radial_basis = MonomialBasis(**radial_hypers)
        else:
            raise NotImplementedError(
                f"{self.radial_basis_name} is not an implemented basis"
                ". Try 'monomial' or 'gto'"
            )

        self.num_ns = self.radial_basis.get_num_radial_functions()
        self.sph_to_cart = spherical_to_cartesian(self.max_angular, self.num_ns)

        if rotation_type not in ["quaternion", "matrix"]:
            raise ValueError(
                "We have only implemented transforming quaternions (`quaternion`) and rotation matrices (`matrix`)."
            )
        elif rotation_type == "quaternion":
            self.rotation_maker = lambda q: Rotation.from_quat([*q[1:], q[0]])
            warnings.warn(
                "In quaternion mode, quaternions are assumed to be in (w,x,y,z) format."
            )
        else:
            self.rotation_maker = Rotation.from_matrix

        self.rotation_key = rotation_key

    def transform(self, frames, show_progress=False, normalize=True, rust_moments=True):
        """Computes features and gradients for frames

        Computes the features and (if compute_gradients == True) gradients
        for all the provided frames. The features and gradients are stored in
        features and feature_gradients attribute.

        Parameters
        ----------
        frames : ase.Atoms
            List containing all ase.Atoms types
        show_progress : bool
            Show progress bar for frame analysis and feature generation
        normalize : bool
            Whether to perform Lowdin Symmetric Orthonormalization or not.
        rust_moments : bool
            Use the ported rust code, which should result in increased speed. Default = True.
            In the future, once we ensure integrity checks with the original python code,
            this kwarg will be deprecated, and the rust version will always be used.

        Returns
        -------
        None, but stores the projection coefficients and (if desired)
        gradients as arrays as `features` and `features_gradients`.

        """
        self.frames = frames

        # Generate a dictionary to map atomic types to array indices
        # In general, the types are sorted according to atomic number
        # and assigned the array indices 0, 1, 2,...
        # Example: for H2O: H is mapped to 0 and O is mapped to 1.
        num_frames = len(frames)
        types = set()
        self.num_atoms_per_frame = np.zeros((num_frames), int)

        for i, f in enumerate(self.frames):
            self.num_atoms_per_frame[i] = len(f)
            for atom in f:
                types.add(atom.number)

        self.num_atoms_total = sum(self.num_atoms_per_frame)
        types = sorted(types)

        # Define variables determining size of feature vector coming from frames
        self.num_atoms_per_frame = np.array([len(frame) for frame in frames])

        num_particle_types = len(types)

        # Initialize arrays in which to store all features
        self.feature_gradients = 0

        frame_generator = tqdm(
            self.frames, disable=(not show_progress), desc="Computing neighborlist"
        )

        self.frame_to_global_atom_idx = np.zeros((num_frames), int)
        for n in range(1, num_frames):
            self.frame_to_global_atom_idx[n] = (
                self.num_atoms_per_frame[n - 1] + self.frame_to_global_atom_idx[n - 1]
            )

        rotation_matrices = np.zeros((self.num_atoms_total, 3, 3))
        ellipsoid_lengths = np.zeros((self.num_atoms_total, 3))

        for i in range(num_frames):
            for j in range(self.num_atoms_per_frame[i]):
                j_global = self.frame_to_global_atom_idx[i] + j
                if self.rotation_key in frames[i].arrays:
                    rotation_matrices[j_global] = self.rotation_maker(
                        frames[i].arrays[self.rotation_key][j]
                    ).as_matrix()
                else:
                    warnings.warn(
                        f"Frame {i} does not have rotations stored, this may cause errors down the line."
                    )

                ellipsoid_lengths[j_global] = [
                    frames[i].arrays["c_diameter[1]"][j] / 2,
                    frames[i].arrays["c_diameter[2]"][j] / 2,
                    frames[i].arrays["c_diameter[3]"][j] / 2,
                ]

        self.nl = NeighborList(
            cutoff=self.cutoff_radius,
            full_neighbor_list=True,
            self_pairs=(not self.subtract_center_contribution),
        ).compute(frame_generator)

        pairwise_ellip_feat = pairwise_ellip_expansion(
            self.max_angular,
            self.nl,
            types,
            self.frame_to_global_atom_idx,
            rotation_matrices,
            ellipsoid_lengths,
            self.sph_to_cart,
            self.radial_basis,
            show_progress,
            rust_moments=rust_moments,
        )

        features = contract_pairwise_feat(pairwise_ellip_feat, types, show_progress)
        if normalize:
            normalized_features = self.radial_basis.orthonormalize_basis(features)
            return normalized_features
        else:
            return features

    def power_spectrum(self, frames, sum_over_samples=True):
        """Function to compute the power spectrum of AniSOAP

        computes the power spectrum of AniSOAP with the inputs of AniSOAP hyperparameters
        and ellipsoidal frames using ellipsoidal density projection. It checks if
        each ellipsoidal frame contains all required attributes and processes
        Clebsch-Gordan coefficients for the angular component of the AniSOAP descriptors.

        Parameters
        ----------

        frames: list
            A list of ellipsoidal frames, where each frame contains attributes:
            'c_diameter[1]', 'c_diameter[2]', 'c_diameter[3]', 'c_q', 'positions', and 'numbers'.
            It only accepts c_q for the angular attribute of each frame.

        sum_over_sample: bool
            A function that returns the sum of coefficients of the frames in the sample.

        Returns
        -------
        x_asoap_raw when kwarg sum_over_samples=True or mvg_nu2 when sum_over_samples=False:
        x_asoap_raw: A 2-dimensional np.array with shape (n_samples, n_features). This AniSOAP power spectrum aggregates (sums) over each sample.
        mvg_nu2: a TensorMap of unaggregated power spectrum features.

        """

        # Initialize the Clebsch Gordan calculator for the angular component.
        mycg = ClebschGordanReal(self.max_angular)

        # Checks that the sample's first frame is not empty
        if frames[0].arrays is None:
            raise ValueError("frames cannot be none")
        required_attributes = [
            "c_diameter[1]",
            "c_diameter[2]",
            "c_diameter[3]",
            "c_q",
            "positions",
            "numbers",
        ]

        # Checks if the sample contains all necessary information for computation of power spectrum
        for index, frame in enumerate(frames):
            array = frame.arrays
            for attr in required_attributes:
                if attr not in array:
                    raise ValueError(
                        f"frame at index {index} is missing a required attribute '{attr}'"
                    )
                if "quaternion" in array:
                    raise ValueError(f"frame should contain c_q rather than quaternion")

        mvg_coeffs = self.transform(frames, show_progress=True)
        mvg_nu1 = standardize_keys(mvg_coeffs)

        # Combines the mvg_nu1 with itself using the Clebsch-Gordan coefficients.
        # This combines the angular and radial components of the sample.
        mvg_nu2 = cg_combine(
            mvg_nu1,
            mvg_nu1,
            clebsch_gordan=mycg,
            lcut=0,
            other_keys_match=["types_center"],
        )

        # If sum_over_samples = True, it returns simplified form of coefficients with fewer dimensions in the TensorMap for subsequent visualization.
        # If not, it returns raw numerical data of coefficients in mvg_nu2 TensorMap
        if sum_over_samples:
            x_asoap_raw = metatensor.sum_over_samples(mvg_nu2, sample_names="center")
            x_asoap_raw = x_asoap_raw.block().values.squeeze()
            return x_asoap_raw
        else:
            return mvg_nu2
