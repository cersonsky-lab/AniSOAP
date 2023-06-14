import numpy as np
from scipy.special import gamma
from equistore.core import (
    Labels,
    TensorBlock,
    TensorMap,
)
import warnings

class RadialBasis:
    """
    Class for precomputing and storing all results related to the radial basis.
    This helps to keep a cleaner main code by avoiding if-else clauses
    related to the radial basis.

    TODO: In the long run, this class would precompute quantities like
    the normalization factors or orthonormalization matrix for the
    radial basis.
    """

    def __init__(self, radial_basis, max_angular, **hypers):
        # Store all inputs into internal variables
        self.radial_basis = radial_basis
        self.max_angular = max_angular
        self.hypers = hypers
        if self.radial_basis not in ["monomial", "gto"]:
            raise ValueError(f"{self.radial_basis} is not an implemented basis.")

        # As part of the initialization, compute the number of radial basis
        # functions, nmax, for each angular frequency l.
        self.num_radial_functions = []
        for l in range(max_angular + 1):
            num_n = (max_angular - l) // 2 + 1
            self.num_radial_functions.append(num_n)

    # Get number of radial functions
    def get_num_radial_functions(self):
        return self.num_radial_functions

    # For each particle pair (i,j), we are provided with the three quantities
    # that completely define the Gaussian distribution, namely
    # the pair distance r_ij, the rotation matrix specifying the orientation
    # of particle j's ellipsoid, as well the the three lengths of the
    # principal axes.
    # Combined with the choice of radial basis, these completely specify
    # the mathematical problem, namely the integral that needs to be
    # computed, which will be of the form
    # integral gaussian(x,y,z) * polynomial(x,y,z) dx dy dz
    # This function deals with the Gaussian part, which is specified
    # by a precision matrix (inverse of covariance) and its center.
    # The current function computes the covariance matrix and the center
    # for the provided parameters as well as choice of radial basis.
    def compute_gaussian_parameters(self, r_ij, lengths, rotation_matrix):
        # Initialization
        center = r_ij
        diag = np.diag(1 / lengths**2)
        precision = rotation_matrix @ diag @ rotation_matrix.T

        # GTO basis with uniform Gaussian width in the basis functions
        if self.radial_basis == "gto":
            sigma = self.hypers["radial_gaussian_width"]
            precision += np.eye(3) / sigma**2
            center -= 1 / sigma**2 * np.linalg.solve(precision, r_ij)

        return precision, center

    def normalize_basis(self, features: TensorMap):
        """
        In-place multiply each value within each block of the features TensorMap by the appropriate normalization value.
        These normalization values are given in equation 10 here: https://pubs.aip.org/aip/jcp/article/154/11/114109/315400/Efficient-implementation-of-atom-density
        and is implemented in librascal here: https://github.com/lab-cosmo/librascal/blob/a4ffbc772ad97ce6cbe9b46900660236b94d2ee2/bindings/rascal/utils/radial_basis.py#L100
        This normalization scales down the GTO portion appropriately, but I'm still unsure what the normalizationr represents.
        i.e. I'm not sure if the normalization ensures that the integral from 0 to inf = 1, or if the integral from 0 to inf
        of the GTO^2 = 1, or something else.

        Parameters:
            features: A TensorMap whose blocks' values we wish to normalize. Note that features is modified in place, so a
            copy of features must be made before the function if you wish to retain the unnormalized values.
            radial_basis: An instance of RadialBasis

        Returns:
            normalized_features: features containing values multiplied by proper normalization factors.
        """
        normalized_features = features.copy()
        radial_basis_name = self.radial_basis
        sigma = self.hypers["radial_gaussian_width"]
        if radial_basis_name != "gto":
            warnings.warn("Have not implemented normalization for non-gto basis, will return original values")
            return features
        for l, block in enumerate(normalized_features.blocks()):
            for k, property in enumerate(block.properties):
                n = property[0]
                l_2n = l + 2 * n
                N = np.sqrt(2 / (sigma ** (2 * l_2n + 3) * gamma(l_2n + 1.5)))
                block.values[:, :, k] *= N
        
        return normalized_features
