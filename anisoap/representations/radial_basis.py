import warnings

import numpy as np
import scipy.linalg
from metatensor import TensorMap
from scipy.special import gamma


def inverse_matrix_sqrt(matrix: np.array):
    """
    Returns the inverse matrix square root.
    The inverse square root of the overlap matrix (or slices of the overlap matrix) yields the
    orthonormalization matrix
    Args:
        matrix: np.array
            Symmetric square matrix to find the inverse square root of

    Returns:
        inverse_sqrt_matrix: S^{-1/2}

    """
    if not np.allclose(matrix, matrix.conjugate().T):
        raise ValueError("Matrix is not hermitian")
    eva, eve = np.linalg.eigh(matrix)

    if (eva < 0).any():
        raise ValueError(
            "Matrix is not positive semidefinite. Check that a valid gram matrix is passed."
        )
    return eve @ np.diag(1 / np.sqrt(eva)) @ eve.T


def gto_square_norm(n, sigma):
    """
    Compute the square norm of GTOs (inner product of itself over R^3).
    An unnormalized GTO of order n is \phi_n = r^n * e^{-r^2/(2*\sigma^2)}
    The square norm of the unnormalized GTO has an analytic solution:
    <\phi_n | \phi_n> = \int_0^\infty dr r^2 |\phi_n|^2 = 1/2 * \sigma^{2n+3} * \Gamma(n+3/2)
    Args:
        n: order of the GTO
        sigma: width of the GTO

    Returns:
        square norm: The square norm of the unnormalized GTO
    """
    return 0.5 * sigma ** (2 * n + 3) * gamma(n + 1.5)


def gto_prefactor(n, sigma):
    """
    Computes the normalization prefactor of an unnormalized GTO.
    This prefactor is simply 1/sqrt(square_norm_area).
    Scaling a GTO by this prefactor will ensure that the GTO has square norm equal to 1.
    Args:
        n: order of GTO
        sigma: width of GTO

    Returns:
        N: normalization constant

    """
    return np.sqrt(1 / gto_square_norm(n, sigma))


def gto_overlap(n, m, sigma_n, sigma_m):
    """
    Compute overlap of two *normalized* GTOs
    Note that the overlap of two GTOs can be modeled as the square norm of one GTO, with an effective
    n and sigma. All we need to do is to calculate those effective parameters, then compute the normalization.
    <\phi_n, \phi_m> = \int_0^\infty dr r^2 r^n * e^{-r^2/(2*\sigma_n^2) * r^m * e^{-r^2/(2*\sigma_m^2)
    = \int_0^\infty dr r^2 |r^{(n+m)/2} * e^{-r^2/4 * (1/\sigma_n^2 + 1/\sigma_m^2)}|^2
    = \int_0^\infty dr r^2 r^n_{eff} * e^{-r^2/(2*\sigma_{eff}^2)
    prefactor.
    ---Arguments---
    n: order of the first GTO
    m: order of the second GTO
    sigma_n: sigma parameter of the first GTO
    sigma_m: sigma parameter of the second GTO

    ---Returns---
    S: overlap of the two normalized GTOs
    """
    N_n = gto_prefactor(n, sigma_n)
    N_m = gto_prefactor(m, sigma_m)
    n_eff = (n + m) / 2
    sigma_eff = np.sqrt(2 * sigma_n**2 * sigma_m**2 / (sigma_n**2 + sigma_m**2))
    return N_n * N_m * gto_square_norm(n_eff, sigma_eff)


class RadialBasis:
    """
    Class for precomputing and storing all results related to the radial basis.
    This helps to keep a cleaner main code by avoiding if-else clauses
    related to the radial basis.

    Code relating to GTO orthonormalization is heavily inspired by work done in librascal, specifically this
    codebase here: https://github.com/lab-cosmo/librascal/blob/8405cbdc0b5c72a5f0b0c93593100dde348bb95f/bindings/rascal/utils/radial_basis.py

    """

    def __init__(self, radial_basis, max_angular, cutoff_radius, num_radial=None, **hypers):
        # Store all inputs into internal variables
        self.radial_basis = radial_basis
        self.max_angular = max_angular
        self.cutoff_radius = cutoff_radius
        self.hypers = hypers
        if self.radial_basis not in ["monomial", "gto"]:
            raise ValueError(f"{self.radial_basis} is not an implemented basis.")

        # As part of the initialization, compute the number of radial basis
        # functions, nmax, for each angular frequency l.
        self.num_radial_functions = []
        for l in range(max_angular + 1):
            if num_radial is None:
                num_n = (max_angular - l) // 2 + 1
                self.num_radial_functions.append(num_n)
            elif isinstance(num_radial, list):
                if len(num_radial) < l:
                    raise ValueError(
                        "If you specify a list of number of radial components, this list must be of length {}. Received {}.".format(
                            max_angular + 1, len(num_radial)
                        )
                    )
                if not isinstance(num_radial[l], int):
                    raise ValueError("`num_radial` must be None, int, or list of int")
                self.num_radial_functions.append(num_radial[l])
            elif isinstance(num_radial, int):
                self.num_radial_functions.append(num_radial)
            else:
                raise ValueError("`num_radial` must be None, int, or list of int")

        # As part of the initialization, compute the orthonormalization matrix for GTOs
        # If we are using the monomial basis, set self.overlap_matrix equal to None
        self.overlap_matrix = None
        if self.radial_basis == "gto":
            self.overlap_matrix = self.calc_gto_overlap_matrix()

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
    def compute_gaussian_parameters(self, r_ij, lengths, rotation_matrix, radial_gaussian_width=None):
        # Initialization
        center = r_ij
        diag = np.diag(1 / lengths**2)
        precision = rotation_matrix @ diag @ rotation_matrix.T

        # GTO basis with uniform Gaussian width in the basis functions
        if self.radial_basis == "gto":
            sigma = radial_gaussian_width
            if radial_gaussian_width is None:
                sigma = self.hypers["radial_gaussian_width"]
            precision += np.eye(3) / sigma**2
            center -= 1 / sigma**2 * np.linalg.solve(precision, r_ij)

        return precision, center

    def calc_gto_overlap_matrix(self):
        """
        Computes the overlap matrix for GTOs.
        The overlap matrix is a Gram matrix whose entries are the overlap: S_{ij} = \int_0^\infty dr r^2 phi_i phi_j
        The overlap has an analytic solution (see above functions).
        The overlap matrix is the first step to generating an orthonormal basis set of functions (Lodwin Symmetric
        Orthonormalization). The actual orthonormalization matrix cannot be fully precomputed because each tensor
        block use a different set of GTOs. Hence, we precompute the full overlap matrix of dim l_max, and while
        orthonormalizing each tensor block, we generate the respective orthonormal matrices from slices of the full
        overlap matrix.

        Returns:
            S: 2D array. The overlap matrix
        """
        # Consequence of the floor divide used to compute self.num_radial_functions
        max_deg = self.max_angular + 1
        n_grid = np.arange(max_deg)
        sigma = self.hypers["radial_gaussian_width"]
        sigma_grid = np.ones(max_deg) * sigma
        S = gto_overlap(
            n_grid[:, np.newaxis],
            n_grid[np.newaxis, :],
            sigma_grid[:, np.newaxis],
            sigma_grid[np.newaxis, :],
        )
        return S

    def orthonormalize_basis(self, features: TensorMap):
        """
        Apply an in-place orthonormalization on the features, using Lodwin Symmetric Orthonormalization.
        Each block in the features TensorMap uses a GTO set of l + 2n, so we must take the appropriate slices of
        the overlap matrix to compute the orthonormalization matrix.
        An instructive example of Lodwin Symmetric Orthonormalization of a 2-element basis set is found here:
        https://booksite.elsevier.com/9780444594365/downloads/16755_10030.pdf

        Parameters:
            features: A TensorMap whose blocks' values we wish to orthonormalize. Note that features is modified in place, so a
            copy of features must be made before the function if you wish to retain the unnormalized values.
            radial_basis: An instance of RadialBasis

        Returns:
            normalized_features: features containing values multiplied by proper normalization factors.
        """
        # In-place modification.
        radial_basis_name = self.radial_basis
        if radial_basis_name != "gto":
            warnings.warn(
                f"Normalization has not been implemented for the {radial_basis_name} basis, and features will not be normalized.",
                UserWarning,
            )
            return features
        for label, block in features.items():
            l = label["angular_channel"]
            n_arr = block.properties["n"].flatten()
            l_2n_arr = l + 2 * n_arr
            # normalize all the GTOs by the appropriate prefactor first, since the overlap matrix is in terms of
            # normalized GTOs
            # Ensure that each gto width is the cutoff_radius * sqrt(n) / nmax. If n < 1, then take n = 1.
            nmax = n_arr[-1] + 1
            sigma_arr = []
            for n in n_arr:
                if n < 1:
                    n = 1
                sigma_arr.append(self.cutoff_radius * np.sqrt(n) / nmax)

            sigma_arr = np.array(sigma_arr)
            prefactor_arr = gto_prefactor(
                l_2n_arr, sigma_arr
            )
            prefactor_arr = gto_prefactor(l_2n_arr, self.hypers["radial_gaussian_width"])
            block.values[:, :, :] = block.values[:, :, :] * prefactor_arr

            gto_overlap_matrix_slice = self.overlap_matrix[l_2n_arr, :][:, l_2n_arr]
            orthonormalization_matrix = inverse_matrix_sqrt(gto_overlap_matrix_slice)
            block.values[:, :, :] = np.einsum(
                "ijk,kl->ijl", block.values, orthonormalization_matrix
            )

        return features
