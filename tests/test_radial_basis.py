import numpy as np
import pytest
from numpy.testing import assert_allclose

# internal imports
from anisoap.representations import RadialBasis, radial_basis
from anisoap.utils import quaternion_to_rotation_matrix


class TestNumberOfRadialFunctions:
    """
    Test that the number of radial basis functions is correct.
    """

    def test_radial_functions_n5(self):
        basis_gto = RadialBasis(radial_basis="monomial", max_angular=5)
        num_ns = basis_gto.get_num_radial_functions()

        # Compare against exact results
        num_ns_exact = [3, 3, 2, 2, 1, 1]
        assert len(num_ns) == len(num_ns_exact)
        for l, num in enumerate(num_ns):
            assert num == num_ns_exact[l]

    def test_radial_functions_n6(self):
        basis_gto = RadialBasis(radial_basis="monomial", max_angular=6)
        num_ns = basis_gto.get_num_radial_functions()

        # Compare against exact results
        num_ns_exact = [4, 3, 3, 2, 2, 1, 1]
        assert len(num_ns) == len(num_ns_exact)
        for l, num in enumerate(num_ns):
            assert num == num_ns_exact[l]


class TestGaussianParameters:
    """
    Test that the two quantities determining a Gaussian distribution, namely
    the precision matrix (inverse of covariance matrix) and center are
    computed correctly in the radial basis class.
    """

    # Generate a random set of orientations and principal axis lengths
    np.random.seed(3923)
    num_random = 3
    quaternions = np.random.normal(size=(4, num_random))
    quaternions /= np.linalg.norm(quaternions, axis=0)
    rotation_matrices = np.zeros((num_random, 3, 3))
    for i, quat in enumerate(quaternions.T):
        rotation_matrices[i] = quaternion_to_rotation_matrix(quat)
    lengths = np.random.uniform(low=0.5, high=5.0, size=((num_random, 3)))
    r_ijs = np.random.normal(scale=5.0, size=((num_random, 3)))

    # For large sigmas, the primitive gto basis should reduce to the
    # monomial basis.
    sigmas_large = np.geomspace(1e10, 1e18, 4)

    @pytest.mark.parametrize("sigma", sigmas_large)
    @pytest.mark.parametrize("r_ij", r_ijs)
    @pytest.mark.parametrize("lengths", lengths)
    @pytest.mark.parametrize("rotation_matrix", rotation_matrices)
    def test_limit_large_sigma(self, sigma, r_ij, lengths, rotation_matrix):
        # Initialize the classes
        basis_mon = RadialBasis(radial_basis="monomial", max_angular=2)
        basis_gto = RadialBasis(
            radial_basis="gto", radial_gaussian_width=sigma, max_angular=2
        )

        # Get the center and precision matrix
        hypers = {}
        hypers["r_ij"] = r_ij
        hypers["lengths"] = lengths
        hypers["rotation_matrix"] = rotation_matrix
        prec_mon, center_mon = basis_mon.compute_gaussian_parameters(**hypers)
        prec_gto, center_gto = basis_gto.compute_gaussian_parameters(**hypers)

        # Check that for large sigma, the two are close
        assert_allclose(center_mon, center_gto, rtol=1e-10, atol=1e-15)
        assert_allclose(prec_mon, prec_gto, rtol=1e-10, atol=1e-15)

    # For small sigmas, the precision matrix and center should
    # be dominated by the isotropic part.
    sigmas_small = np.geomspace(1e-10, 1e-18, 4)

    @pytest.mark.parametrize("sigma", sigmas_small)
    @pytest.mark.parametrize("r_ij", r_ijs)
    @pytest.mark.parametrize("lengths", lengths)
    @pytest.mark.parametrize("rotation_matrix", rotation_matrices)
    def test_limit_small_sigma(self, sigma, r_ij, lengths, rotation_matrix):
        # Initialize the class
        basis_gto = RadialBasis(
            radial_basis="gto", radial_gaussian_width=sigma, max_angular=2
        )

        # Get the center and precision matrix
        hypers = {}
        hypers["r_ij"] = r_ij
        hypers["lengths"] = lengths
        hypers["rotation_matrix"] = rotation_matrix
        prec_gto, center_gto = basis_gto.compute_gaussian_parameters(**hypers)

        # Check that for large sigma, the two are close
        prec_ref = np.eye(3) / sigma**2
        center_ref = np.zeros((3,))
        atol = 1e-15 / sigma**2  # largest elements of matrix will be 1/sigma^2
        assert_allclose(center_gto, center_ref, rtol=1e-10, atol=atol)
        assert_allclose(prec_gto, prec_ref, rtol=1e-10, atol=atol)
