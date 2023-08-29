import sys
import warnings
from itertools import product

import numpy as np
from equistore.core import (
    Labels,
    TensorBlock,
    TensorMap,
)
from rascaline import NeighborList
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

from anisoap.representations.radial_basis import RadialBasis
from anisoap.utils.moment_generator import *
from anisoap.utils.spherical_to_cartesian import spherical_to_cartesian

# ADDED
from anisoap.lib import compute_moments


def pairwise_ellip_expansion(
    lmax,
    neighbor_list,
    species,
    frame_to_global_atom_idx,
    rotation_matrices,
    ellipsoid_lengths,
    sph_to_cart,
    radial_basis,
    *,
    version: int = None,
    show_progress=False,
):
    """
    Function to compute the pairwise expansion <anlm|rho_ij> by combining the moments and the spherical to Cartesian
    transformation
    --------------------------------------------------------
    Parameters:
    lmax : int
        Maximum angular

    neighbor_list : Equistore TensorMap
        Full neighborlist with keys (species_1, species_2) enumerating the possible species pairs.
        Each block contains as samples, the atom indices of (first_atom, second_atom) that correspond to the key,
        and block.value is a 3D array of the form (num_samples, num_components,properties), with num_components=3
        corresponding to the (x,y,z) components of the vector from first_atom to second_atom.
        Depending on the cutoff some species pairs may not appear. Self pairs are not present but in PBC,
        pairs between copies of the same atom are accounted for.

    species: list of ints
        List of atomic numbers present across the data frames

    frame_to_global_atom_idx: list of ints
        The length of the list equals the number of frames, each entry enumerating the number of atoms in
        corresponding frame

    rotation_matrices: np.array of dimension ((num_atoms,3,3))

    ellipsoid_lengths: np.array of dimension ((num_atoms,3))

    radial_basis : Instance of the RadialBasis Class
        anisoap.representations.radial_basis.RadialBasis that has been instantiated appropriately with the
        cutoff radius, radial basis type.

    show_progress : bool
        Show progress bar for frame analysis and feature generation
    -----------------------------------------------------------
    Returns:
        An Equistore TensorMap with keys (species_1, species_2, l) where ("species_1", "species_2") is key in the
        neighbor_list and "l" is the angular channel.
        Each block of this tensormap has the same samples as the corresponding block of the neighbor_list.
        block.value is a 3D array of the form (num_samples, num_components, properties) where num_components
        form the 2*l+1 values for the corresponding angular channel and the properties dimension corresponds to
        the radial channel.

    """
    tensorblock_list = []
    keys = np.asarray(neighbor_list.keys, dtype=int)
    keys = [tuple(i) + (l,) for i in keys for l in range(lmax + 1)]
    num_ns = radial_basis.get_num_radial_functions()
    
    if version is None or version >= 2:    # Changes the loop structure
        for (center_species, neighbor_species) in neighbor_list.keys:
            _single_pair_expansion(tensorblock_list, neighbor_list, center_species,
                                   neighbor_species, lmax, frame_to_global_atom_idx,
                                   rotation_matrices, ellipsoid_lengths, radial_basis, 
                                   num_ns, sph_to_cart, show_progress, version=version)
    else:
        for center_species in species:
            for neighbor_species in species:
                if (center_species, neighbor_species) in neighbor_list.keys:
                    _single_pair_expansion(tensorblock_list, neighbor_list, center_species,
                                           neighbor_species, lmax, frame_to_global_atom_idx,
                                           rotation_matrices, ellipsoid_lengths, radial_basis, 
                                           num_ns, sph_to_cart, show_progress, version=version)
                
    pairwise_ellip_feat = TensorMap(
        Labels(
            ["species_center", "species_neighbor", "angular_channel"],
            np.asarray(keys, dtype=np.int32),
        ),
        tensorblock_list,
    )
    return pairwise_ellip_feat

def _single_pair_expansion(tensorblock_list, neighbor_list, center_species,
                           neighbor_species, lmax, frame_to_global_atom_idx,
                           rotation_matrices, ellipsoid_lengths, radial_basis, 
                           num_ns, sph_to_cart, show_progress, *, version):
    """
    A helper function for single iteration inside the for loop in pairwise_ellip_expansion.
    It was factored out to test for v2's performance.

    This function is NOT expected to be used outside of this file.
    Also, body of this function is expected to go back to pairwise_ellip_expansion in the
    final version, as it is not necessary anymore.
    """
    nl_block = neighbor_list.block(
        species_first_atom=center_species,
        species_second_atom=neighbor_species,
    )
                
    # moved from original position
    values_ldict = {l: [] for l in range(lmax + 1)}
    for isample, nl_sample in enumerate(
        tqdm(
            nl_block.samples,
            disable=(not show_progress),
            desc="Iterating samples for ({}, {})".format(
                center_species, neighbor_species
            ),
            leave=False,
        )
    ):
        frame_idx, i, j = (
            nl_sample["structure"],
            nl_sample["first_atom"],
            nl_sample["second_atom"],
        )
                    
        r_ij = np.asarray(
            [
                nl_block.values[isample, 0],
                nl_block.values[isample, 1],
                nl_block.values[isample, 2],
            ]
        ).reshape(
            3,
        )
                    
        # moved from original position
        j_global = frame_to_global_atom_idx[frame_idx] + j
        rot = rotation_matrices[j_global]
        lengths = ellipsoid_lengths[j_global]

        precision, center = radial_basis.compute_gaussian_parameters(
            r_ij, lengths, rot
        )

        if version is None or version >= 1:
            # NOTE: This line was replaced with Rust implementation.
            moments = compute_moments(precision, center, lmax + np.max(num_ns))
        else:
            moments = compute_moments_inefficient_implementation(
                precision, center, maxdeg=lmax + np.max(num_ns)
            )

        for l in range(lmax + 1):
            deg = l + 2 * (num_ns[l] - 1)
            moments_l = moments[: deg + 1, : deg + 1, : deg + 1]
            values_ldict[l].append(np.einsum("mnpqr, pqr->mn", sph_to_cart[l], moments_l))

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
                np.asarray(
                    list(range(num_ns[l])), np.int32).reshape(-1, 1),
            ),
        )
        tensorblock_list.append(block)


def contract_pairwise_feat(pair_ellip_feat, species, show_progress=False):
    """
    Function to sum over the pairwise expansion \sum_{j in a} <anlm|rho_ij> = <anlm|rho_i>
    --------------------------------------------------------
    Parameters:

    pair_ellip_feat : Equistore TensorMap
        TensorMap returned from "pairwise_ellip_expansion()" with keys (species_1, species_2,l) enumerating
        the possible species pairs and the angular channels.

    species: list of ints
        List of atomic numbers present across the data frames

    show_progress : bool
        Show progress bar for frame analysis and feature generation

    -----------------------------------------------------------
    Returns:
        An Equistore TensorMap with keys (species, l) "species" takes the value of the atomic numbers present
        in the dataset and "l" is the angular channel.
        Each block of this tensormap has as samples ("structure", "center") yielding the indices of the frames
        and atoms that correspond to "species" are present.
        block.value is a 3D array of the form (num_samples, num_components, properties) where num_components
        take on the same values as in the pair_ellip_feat_feat.block .  block.properties now has an additional index
        for neighbor_species that corresponds to "a" in <anlm|rho_i>

    """
    ellip_keys = list(
        set([tuple(list(x)[:1] + list(x)[2:]) for x in pair_ellip_feat.keys])
    )
    # Select the unique combinations of pair_ellip_feat.keys["species_center"] and
    # pair_ellip_feat.keys["angular_channel"] to form the keys of the single particle centered feature
    ellip_keys.sort()
    ellip_blocks = []
    property_names = pair_ellip_feat.property_names + [
        "neighbor_species",
    ]

    for key in tqdm(
        ellip_keys, disable=(not show_progress), desc="Iterating tensor block keys"
    ):
        contract_blocks = []
        contract_properties = []
        contract_samples = []
        # these collect the values, properties and samples of the blocks when contracted over neighbor_species.
        # All these lists have as many entries as len(species).

        for ele in tqdm(
            species,
            disable=(not show_progress),
            desc="Iterating neighbor species",
            leave=False,
        ):
            selection = Labels(
                names=["species_neighbor"], values=np.array([[ele]]))
            blockidx = pair_ellip_feat.blocks_matching(selection=selection)
            # indices of the blocks in pair_ellip_feat with neighbor species = ele
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
                # print(key, ele, "skipped") # this block is not found in the pairwise feat
                continue
            assert len(sel_blocks) == 1

            # sel_blocks is the corresponding block in the pairwise feat with the same (species_center, l) and
            # species_neighbor = ele thus there can be only one block corresponding to the triplet (species_center, species_neighbor, l)
            block = sel_blocks[0]

            pair_block_sample = list(
                zip(block.samples["structure"], block.samples["first_atom"])
            )

            # Takes the structure and first atom index from the current pair_block sample. There might be repeated
            # entries here because for example (0,0,1) (0,0,2) might be samples of the pair block (the index of the
            # neighbor atom is changing but for both of these we are keeping (0,0) corresponding to the structure and
            # first atom.

            struct, center = np.unique(block.samples["structure"]), np.unique(
                block.samples["first_atom"]
            )
            possible_block_samples = list(product(struct, center))
            # possible block samples contains all *unique* possible pairwise products between structure and atom index
            # From here we choose the entries that are actually present in the block to form the final sample

            block_samples = []
            block_values = []

            for isample, sample in enumerate(
                tqdm(
                    possible_block_samples,
                    disable=(not show_progress),
                    desc="Finding matching block samples",
                    leave=False,
                )
            ):
                sample_idx = [
                    idx
                    for idx, tup in enumerate(pair_block_sample)
                    if tup[0] == sample[0] and tup[1] == sample[1]
                ]
                # all samples of the pair block that match the current sample
                # in the example above, for sample = (0,0) we would identify sample_idx = [(0,0,1), (0,0,2)]
                if len(sample_idx) == 0:
                    continue
                # print(key, ele, sample, block.samples[sample_idx])
                block_samples.append(sample)
                block_values.append(block.values[sample_idx].sum(axis=0))  # sum over "j"  for given ele

                # block_values has as many entries as samples satisfying (key, neighbor_species=ele).
                # When we iterate over neighbor species, not all (structure, center) would be present
                # Example: (0,0,1) might be present in a block with neighbor_species = 1 but no other pair block
                # ever has (0,0,x) present as a sample- so (0,0) doesn't show up in a block_sample for all ele
                # so in general we have a ragged list of contract_blocks

            contract_blocks.append(block_values)
            contract_samples.append(block_samples)
            contract_properties.append([tuple(p) + (ele,) for p in block.properties])
            # this adds the "ele" (i.e. neighbor_species) to the properties dimension

        #         print(len(contract_samples))
        all_block_samples = sorted(list(set().union(*contract_samples)))
        # Selects the set of samples from all the block_samples we collected by iterating over the neighbor_species
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
        # len(contract_blocks) - adds the additional dimension for the neighbor_species since we accumulated
        # values for each of them as \sum_{j in ele} <|rho_ij>
        #  Thus - all_block_values.shape = (num_final_samples, components_pair, properties_pair, num_species)

        for iele, elem_cont_samples in enumerate(
            tqdm(
                contract_samples,
                disable=(not show_progress),
                desc="Contracting features",
                leave=False,
            )
        ):
            # This effectively loops over the species of the neighbors
            # Now we just need to add the contributions to the final samples and values from this species to the right
            # samples
            nzidx = [
                i
                for i in range(len(all_block_samples))
                if all_block_samples[i] in elem_cont_samples
            ]
            # identifies where the samples that this species contributes to, are present in the final samples
            #             print(species[ib], key, bb, all_block_samples)
            all_block_values[nzidx, :, :, iele] = contract_blocks[iele]

        new_block = TensorBlock(
            values=all_block_values.reshape(
                all_block_values.shape[0], all_block_values.shape[1], -1
            ),
            samples=Labels(
                ["structure", "center"], np.asarray(
                    all_block_samples, np.int32)
            ),
            components=block.components,
            properties=Labels(
                list(property_names),
                np.asarray(np.vstack(contract_properties), np.int32),
            ),
        )

        ellip_blocks.append(new_block)
    ellip = TensorMap(
        Labels(
            ["species_center", "angular_channel"],
            np.asarray(ellip_keys, dtype=np.int32),
        ),
        ellip_blocks,
    )

    return ellip


class EllipsoidalDensityProjection:
    """
    Compute the spherical projection coefficients for a system of ellipsoids
    assuming a multivariate Gaussian density.
    Initialize the calculator using the hyperparameters.
    ----------
    max_angular : int
        Number of angular functions
    radial_basis : str
        The radial basis. Currently implemented are
        'GTO_primitive', 'GTO', 'monomial'.
    compute_gradients : bool
        Compute gradients
    subtract_center_contribution : bool
        Subtract contribution from the central atom.
    rotation_key : string
        Key under which rotations are stored in ase frames arrays
    rotation_type : string
        Type of rotation object being passed. Currently implemented
        are 'quaternion' and 'matrix'\

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
        rotation_key="quaternion",
        rotation_type="quaternion",
    ):
        # Store the input variables
        self.max_angular = max_angular
        self.cutoff_radius = cutoff_radius
        self.compute_gradients = compute_gradients
        self.subtract_center_contribution = subtract_center_contribution
        self.radial_basis_name = radial_basis_name

        # Currently, gradients are not supported
        if compute_gradients:
            raise NotImplementedError(
                "Sorry! Gradients have not yet been implemented"
            )
        
        # Initialize the radial basis class
        if radial_basis_name not in ["monomial", "gto"]:
            raise NotImplementedError(
                f"{self.radial_basis_name} is not an implemented basis"
                ". Try 'monomial' or 'gto'"
            )
        if radial_gaussian_width != None and radial_basis_name != "gto":
            raise ValueError(
                "Gaussian width can only be provided with GTO basis"
            )
        elif radial_gaussian_width is None and radial_basis_name == "gto":
            raise ValueError("Gaussian width must be provided with GTO basis")
        elif type(radial_gaussian_width) == int:
            raise ValueError(
                "radial_gaussian_width is set as an integer, which could cause overflow errors. Pass in float."
            )
        radial_hypers = {}
        radial_hypers["radial_basis"] = radial_basis_name.lower()  # lower case
        radial_hypers["radial_gaussian_width"] = radial_gaussian_width
        radial_hypers["max_angular"] = max_angular
        self.radial_basis = RadialBasis(**radial_hypers)

        self.num_ns = self.radial_basis.get_num_radial_functions()
        self.sph_to_cart = spherical_to_cartesian(
            self.max_angular, self.num_ns)

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

    def transform(self, frames, show_progress=False, normalize=True, *, version: int = None):
        """
        Computes the features and (if compute_gradients == True) gradients
        for all the provided frames. The features and gradients are stored in
        features and feature_gradients attribute.
        Parameters
        ----------
        frames : ase.Atoms
            List containing all ase.Atoms structures
        show_progress : bool
            Show progress bar for frame analysis and feature generation
        normalize: bool
            Whether to perform Lowdin Symmetric Orthonormalization or not. Orthonormalization generally
            leads to better performance. Default: True.
        Returns
        -------
        None, but stores the projection coefficients and (if desired)
        gradients as arrays as `features` and `features_gradients`.
        """
        self.frames = frames
        
        # Generate a dictionary to map atomic species to array indices
        # In general, the species are sorted according to atomic number
        # and assigned the array indices 0, 1, 2,...
        # Example: for H2O: H is mapped to 0 and O is mapped to 1.
        num_frames = len(frames)
        species = set()
        self.num_atoms_per_frame = np.zeros((num_frames), int)

        for i, f in enumerate(self.frames):
            self.num_atoms_per_frame[i] = len(f)
            for atom in f:
                species.add(atom.number)
        
        self.num_atoms_total = sum(self.num_atoms_per_frame)
        species = sorted(species)
        
        # Define variables determining size of feature vector coming from frames
        self.num_atoms_per_frame = np.array([len(frame) for frame in frames])
        
        # Initialize arrays in which to store all features
        self.feature_gradients = 0

        frame_generator = tqdm(
            self.frames, disable=(not show_progress), desc="Computing neighborlist"
        )

        self.frame_to_global_atom_idx = np.zeros((num_frames), int)
        for n in range(1, num_frames):
            self.frame_to_global_atom_idx[n] = (
                self.num_atoms_per_frame[n - 1] +
                self.frame_to_global_atom_idx[n - 1]
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
        
        self.nl = NeighborList(self.cutoff_radius, True, True).compute(frame_generator)
        
        pairwise_ellip_feat = pairwise_ellip_expansion(
            self.max_angular,
            self.nl,
            species,
            self.frame_to_global_atom_idx,
            rotation_matrices,
            ellipsoid_lengths,
            self.sph_to_cart,
            self.radial_basis,
            version=version
        )
        
        features = contract_pairwise_feat(pairwise_ellip_feat, species, show_progress)
        
        if normalize:
            return self.radial_basis.orthonormalize_basis(features)
        else:
            return features
