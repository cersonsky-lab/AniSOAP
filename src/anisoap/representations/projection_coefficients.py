import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    tqdm = (lambda i, **kwargs: i)

from ..utils.moment_generator import compute_moments_inefficient_implementation

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
    def __init__(self,
                 max_angular,
                 radial_basis,
                 compute_gradients=False,
                 subtract_center_contribution=False):

        # Store the input variables
        self.max_angular = max_angular
        self.radial_basis = radial_basis.lower()
        self.compute_gradients = compute_gradients
        self.subtract_center_contribution = subtract_center_contribution

        if self.radial_basis not in ["monomial", "gto", "gto_primitive", "gto_analytical"]:
            raise ValueError(f"{self.radial_basis} is not an implemented basis"
                              ". Try 'monomial', 'GTO' or GTO_primitive.")

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
           self.species_dict.update({atom.symbol: species.index(atom.number) for atom in frame})

        # Define variables determining size of feature vector coming from frames
        self.num_atoms_per_frame = np.array([len(frame) for frame in frames])
        num_atoms_total = np.sum(self.num_atoms_per_frame)
        num_particle_types = len(self.species_dict)
        num_features_total = (self.max_angular+1)**2

        # Initialize arrays in which to store all features
        self.features = np.zeros(num_atoms_total, num_particle_types, num_features_total)
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
        
        # TODO:
        # Compute the features for a single frame
        features = 0.
        '''
        for i in range(num_atoms):
            pos_i = positions[i]
            for j in neighbors(i):
                pos_j = positions[j]
                R, principal_components = get_orientation(j)
                moments = compute_moments(...)
                for n,l,m:
                    feat[n,l,m] += ... 
        '''

        return features