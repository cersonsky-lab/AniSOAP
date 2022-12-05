import numpy as np

from anisoap.utils.spherical_to_cartesian import spherical_to_cartesian

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda i, **kwargs: i

from ..utils import (
    compute_moments_diagonal_inefficient_implementation,
    compute_moments_inefficient_implementation,
    quaternion_to_rotation_matrix,
)
from .radial_basis import RadialBasis


class DensityProjectionCalculator:
    """
    Compute the spherical projection coefficients.
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
    ):

        # Store the input variables
        self.max_angular = max_angular
        self.cutoff_radius = cutoff_radius
        self.compute_gradients = compute_gradients
        self.subtract_center_contribution = subtract_center_contribution
        self.radial_basis_name = radial_basis_name

        # Currently, gradients are not supported
        if compute_gradients:
            raise NotImplementedError("Sorry! Gradients have not yet been implemented")

        # Precompute the spherical to Cartesian transformation
        # coefficients.
        num_ns = []
        for l in range(max_angular + 1):
            num_ns.append(max_angular + 1 - l)
        self.sph_to_cart = spherical_to_cartesian(max_angular, num_ns)

        # Initialize the radial basis class
        if radial_basis_name not in ["monomial", "gto"]:
            raise ValueError(
                f"{self.radial_basis} is not an implemented basis"
                ". Try 'monomial' or 'gto'"
            )
        if radial_gaussian_width != None and radial_basis_name != "gto":
            raise ValueError("Gaussian width can only be provided with GTO basis")
        radial_hypers = {}
        radial_hypers["radial_basis"] = radial_basis_name.lower()  # lower case
        radial_hypers["radial_gaussian_width"] = radial_gaussian_width
        radial_hypers["max_angular"] = max_angular
        self.radial_basis = RadialBasis(**radial_hypers)
        self.num_ns = self.radial_basis.get_num_radial_functions()

    def transform(self, frames, show_progress=False):
        """
        Computes the features and (if compute_gradients == True) gradients
        for all the provided frames. The features and gradients are stored in
        features and feature_gradients attribute.
        Parameters
        ----------
        frames : ase.Atoms
            List containing all ase.Atoms structures
        show_progress : bool
            Show progress bar for frame analysis
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
        species = set()
        for frame in frames:
            for atom in frame:
                species.add(atom.number)
        species = sorted(species)
        self.species_dict = {}
        for frame in frames:
            # Get atomic species in dataset
            self.species_dict.update(
                {atom.symbol: species.index(atom.number) for atom in frame}
            )

        # Define variables determining size of feature vector coming from frames
        self.num_atoms_per_frame = np.array([len(frame) for frame in frames])
        num_atoms_total = np.sum(self.num_atoms_per_frame)
        num_particle_types = len(self.species_dict)
        num_features_total = (self.max_angular + 1) ** 2

        # Initialize arrays in which to store all features
        self.features = np.zeros(
            num_atoms_total, num_particle_types, num_features_total
        )
        self.feature_gradients = 0

        if show_progress:
            frame_generator = tqdm(self.frames)
        else:
            frame_generator = self.frames

        for i_frame, frame in enumerate(frame_generator):
            number_of_atoms = self.num_atoms_per_frame[i_frame]
            results = self._transform_single_frame(frame)

    def _transform_single_frame(self, frame):
        """
        Compute features for single frame and return to the transform()
        method which loops over all structures to obtain the complete
        vector for all environments.
        """
        ###
        # Initialization
        ###
        # Define useful shortcuts
        lmax = self.max_angular
        num_atoms = len(frame)
        num_chem_species = len(self.species_dict)
        iterator_species = np.zeros(num_atoms, dtype=int)
        for i, symbol in enumerate(frame.get_chemical_symbols()):
            iterator_species[i] = self.species_dict[symbol]

        # Get the arrays with all
        # TODO: update with correct expressions
        positions = np.zeros((num_atoms, 3))
        quaternions = np.zeros((num_atoms, 4))
        ellipsoid_lengths = np.zeros((num_atoms, 3))

        # Convert quaternions to rotation matrices
        rotation_matrices = np.zeros((num_atoms, 3, 3))
        for i, quat in enumerate(quaternions):
            rotation_matrices[i] = quaternion_to_rotation_matrix(quat)

        # Generate neighbor list
        # TODO: change this to proper neighbor list
        neighbors = []
        for i in range(num_atoms):
            # for now, just treat every atom as a neighbor
            # of itself + the first two atoms in the structure
            neighbors.append([0, 1, i])

        # Compute the features for a single frame
        features = []
        for l in range(lmax + 1):
            features.append(np.zeros((3, 3)))

        for i in range(num_atoms):
            pos_i = positions[i]
            for j in neighbors[i]:
                # Obtain the position and orientation defining
                # the neighbor particle j
                r_ij = pos_i - positions[j]
                rot = rotation_matrices[j]
                lengths = ellipsoid_lengths[j]

                # Compute the moments
                # The moments have shape ((maxdeg+1, maxdeg+1, maxdeg+1))
                precision, center = self.radial_basis.compute_gaussian_parameters(
                    r_ij, lengths, rot
                )
                moments = compute_moments_inefficient_implementation(
                    precision, center, maxdeg=lmax
                )

                for l in range(lmax + 1):
                    features[l] = np.einsum(
                        "mnpqr, pqr->mn", self.sph_to_cart[l], moments
                    )

        return features
