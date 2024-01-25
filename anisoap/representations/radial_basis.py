import warnings

import numpy as np
from metatensor import TensorMap
from scipy.special import gamma


def inverse_matrix_sqrt(matrix: np.array, rcond=1e-8, tol=1e-3):
    r"""Returns the inverse matrix square root.

    The inverse square root of the overlap matrix (or slices of the overlap matrix) 
    yields the orthonormalization matrix

    Parameters
    ----------
    matrix : np.array
        Symmetric square matrix to find the inverse square root of
    rcond: float
        Lower bound for eigenvalues for inverse square root
    tol: float
        Tolerance for differences between original matrix and reconstruction via
        inverse square root

    Returns
    -------
    inverse_sqrt_matrix 
        :math:`S^{-1/2}`

    """
    if not np.allclose(matrix, matrix.conjugate().T):
        raise ValueError("Matrix is not hermitian")
    eva, eve = np.linalg.eigh(matrix)
    eve = eve[:, eva > rcond]
    eva = eva[eva > rcond]

    result = eve @ np.diag(1 / np.sqrt(eva)) @ eve.T

    # Do quick test to make sure inverse square of the inverse matrix sqrt succeeded
    # This should succed for most cases (e.g. GTO orders up to 100), if not the matrix likely wasn't a gram matrix to start with.
    matrix2 = np.linalg.pinv(result @ result)
    if np.linalg.norm(matrix - matrix2) > tol:
        raise ValueError(
            f"Incurred Numerical Imprecision {np.linalg.norm(matrix-matrix2)= :.3f}"
        )
    return result


def gto_square_norm(n, sigma):
    r"""Compute the square norm of GTOs (inner product of itself over :math:`R^3`).

    An unnormalized GTO of order n is :math:`\phi_n = r^n  e^{-r^2/(2*\sigma^2)}`
    The square norm of the unnormalized GTO has an analytic solution:

    .. math:: 

        \braket{\phi_n | \phi_n} &= \int_0^\infty dr \, r^2 \lvert\phi_n\rvert^2 \\
                                 &=  \frac{1}{2} \sigma^{2n+3} \Gamma(n+\frac{3}{2})

    This function uses the above expression.

    Parameters
    ----------
    n 
        order of the GTO
    sigma 
        width of the GTO

    Returns
    -------
    float 
        The square norm of the unnormalized GTO

    """
    return 0.5 * sigma ** (2 * n + 3) * gamma(n + 1.5)


def gto_prefactor(n, sigma):
    """Computes the normalization prefactor of an unnormalized GTO.

    This prefactor is simply :math:`\\frac{1}{\\sqrt(\\text{square_norm_area)}}`.
    Scaling a GTO by this prefactor will ensure that the GTO has square norm 
    equal to 1.

    Parameters
    ----------
    n 
        order of GTO
    sigma 
        width of GTO

    Returns
    -------
    float 
        The normalization constant

    """
    return np.sqrt(1 / gto_square_norm(n, sigma))

def gto_overlap(n, m, sigma_n, sigma_m):
    r"""Compute overlap of two *normalized* GTOs

    Note that the overlap of two GTOs can be modeled as the square norm of one 
    GTO, with an effective :math:`n` and :math:`\sigma`. All we need to do is to 
    calculate those effective parameters, then compute the normalization prefactor.
    
    .. math::

        \langle \phi_n, \phi_m \rangle &= \int_0^\infty dr \, r^2 r^n e^{-r^2/(2\sigma_n^2)} \, r^m  e^{-r^2/(2\sigma_m^2)} \\
                                       &= \int_0^\infty dr \, r^2 \lvert r^{(n+m)/2} e^{-r^2/4 (1/\sigma_n^2 + 1/\sigma_m^2)}\rvert^2 \\
                                       &= \int_0^\infty dr \, r^2 r^n_\text{eff} e^{-r^2/(2\sigma_\text{eff}^2)}


    Parameters
    ----------
    n 
        order of the first GTO
    m 
        order of the second GTO
    sigma_n 
        sigma parameter of the first GTO
    sigma_m 
        sigma parameter of the second GTO

    Returns
    -------
    float 
        overlap of the two normalized GTOs

    """
    N_n = gto_prefactor(n, sigma_n)
    N_m = gto_prefactor(m, sigma_m)
    n_eff = (n + m) / 2
    sigma_eff = np.sqrt(2 * sigma_n**2 * sigma_m**2 / (sigma_n**2 + sigma_m**2))
    return N_n * N_m * gto_square_norm(n_eff, sigma_eff)


class RadialBasis:
    """Class for precomputing and storing all results related to the radial basis.

    This helps to keep a cleaner main code by avoiding if-else clauses
    related to the radial basis.

    Code relating to GTO orthonormalization is heavily inspired by work done in 
    librascal, specifically the codebase found 
    `here <https://github.com/lab-cosmo/librascal/blob/8405cbdc0b5c72a5f0b0c93593100dde348bb95f/bindings/rascal/utils/radial_basis.py>`_

    """

    def __init__(
        self,
        radial_basis,
        max_angular,
        cutoff_radius,
        max_radial=None,
        rcond=1e-8,
        tol=1e-3,
        **hypers,
    ):
        # Store all inputs into internal variables
        self.radial_basis = radial_basis
        self.max_angular = max_angular
        self.cutoff_radius = cutoff_radius
        self.hypers = hypers
        self.rcond = rcond
        self.tol = tol

        if self.radial_basis not in ["monomial", "gto"]:
            raise ValueError(f"{self.radial_basis} is not an implemented basis.")

        # As part of the initialization, compute the number of radial basis
        # functions, num_n, for each angular frequency l.
        # If nmax is given, num_n = nmax + 1 (n ranges from 0 to nmax)
        self.num_radial_functions = []
        for l in range(max_angular + 1):
            if max_radial is None:
                num_n = (max_angular - l) // 2 + 1
                self.num_radial_functions.append(num_n)
            elif isinstance(max_radial, list):
                if len(max_radial) <= l:
                    raise ValueError(
                        "If you specify a list of number of radial components, this list must be of length {}. Received {}.".format(
                            max_angular + 1, len(max_radial)
                        )
                    )
                if not isinstance(max_radial[l], int):
                    raise ValueError("`max_radial` must be None, int, or list of int")
                self.num_radial_functions.append(max_radial[l] + 1)
            elif isinstance(max_radial, int):
                self.num_radial_functions.append(max_radial + 1)
            else:
                raise ValueError("`max_radial` must be None, int, or list of int")

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

    def calc_gto_overlap_matrix(self):
        """Computes the overlap matrix for GTOs.

        The overlap matrix is a Gram matrix whose entries are the overlap: 

        .. math::

            S_{ij} = \\int_0^\\infty dr \\, r^2 \\phi_i \\phi_j

        The overlap has an analytic solution (see above functions).

        The overlap matrix is the first step to generating an orthonormal basis 
        set of functions (Lodwin Symmetric Orthonormalization). The actual 
        orthonormalization matrix cannot be fully precomputed because each tensor
        block uses a different set of GTOs. Hence, we precompute the full overlap 
        matrix of dim l_max, and while orthonormalizing each tensor block, we 
        generate the respective orthonormal matrices from slices of the full
        overlap matrix.

        Returns
        -------
        2D array 
            The overlap matrix

        """
        max_deg = np.max(
            np.arange(self.max_angular + 1) + 2 * np.array(self.num_radial_functions)
        )
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
        """Applies in-place orthonormalization on the features.

        Apply an in-place orthonormalization on the features, using Lodwin Symmetric 
        Orthonormalization. Each block in the features TensorMap uses a GTO set 
        of l + 2n, so we must take the appropriate slices of the overlap matrix 
        to compute the orthonormalization matrix. An instructive example of Lodwin 
        Symmetric Orthonormalization of a 2-element basis set is found here:
        https://booksite.elsevier.com/9780444594365/downloads/16755_10030.pdf

        Parameters
        ----------
        features : TensorMap 
            contains blocks whose values we wish to orthonormalize. Note that 
            features is modified in place, so a copy of features must be made 
            before the function if you wish to retain the unnormalized values.
        radial_basis : RadialBasis

        Returns
        -------
        TensorMap
            features containing values multiplied by normalization factors.

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
            # Each block's `properties` dimension contains radial channels for each neighbor species
            # Hence we have to iterate through each neighbor species and orthonormalize the block in subblocks
            # Each subblock is indexed using the neighbor_mask boolean array.
            neighbors = np.unique(block.properties["neighbor_species"])
            for neighbor in neighbors:
                l = label["angular_channel"]
                neighbor_mask = block.properties["neighbor_species"] == neighbor
                n_arr = block.properties["n"][neighbor_mask].flatten()
                l_2n_arr = l + 2 * n_arr
                # normalize all the GTOs by the appropriate prefactor first, since the overlap matrix is in terms of
                # normalized GTOs
                prefactor_arr = gto_prefactor(
                    l_2n_arr, self.hypers["radial_gaussian_width"]
                )
                block.values[:, :, neighbor_mask] *= prefactor_arr

                gto_overlap_matrix_slice = self.overlap_matrix[l_2n_arr, :][:, l_2n_arr]
                orthonormalization_matrix = inverse_matrix_sqrt(
                    gto_overlap_matrix_slice, self.rcond, self.tol
                )
                block.values[:, :, neighbor_mask] = np.einsum(
                    "ijk,kl->ijl",
                    block.values[:, :, neighbor_mask],
                    orthonormalization_matrix,
                )

        return features
