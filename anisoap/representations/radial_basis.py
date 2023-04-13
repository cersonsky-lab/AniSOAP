import numpy as np
from tqdm import tqdm 

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
        pbar = tqdm(total=max_angular+1)
        for l in range(max_angular + 1):
            num_n = (max_angular - l) // 2 + 1
            pbar.set_description("Computing Radial Basis".format(l))
            pbar.update(1)
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
