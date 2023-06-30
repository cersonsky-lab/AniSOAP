import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

# internal imports
from anisoap.representations import RadialBasis, radial_basis


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
        rotation_matrices[i] = Rotation.from_quat(quat).as_matrix()
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


class TestGTOUtils:
    # Create a list of semipositive definite matrices (spd), seminegative definite matrices (snd), and
    # nonsymmetric matrices for testing

    spd_matrices = []
    snd_matrices = []
    nonsym_matrices = []
    for _ in range(100):
        dim = np.random.randint(2, 100)
        A = np.random.rand(dim, dim)
        spd = A @ A.T
        spd_matrices.append(spd)
        snd_matrices.append(-spd)
        nonsym_matrices.append(np.random.random(size=(dim, dim)))

    num_trials = 100
    basis_sizes = np.random.randint(2, 15, num_trials)

    @pytest.mark.parametrize("spd", spd_matrices)
    def test_spd_inverse_sqrt_no_exceptions(self, spd):
        # Assert that exception is never raised for semipositive definite matrices
        try:
            radial_basis.inverse_matrix_sqrt(spd)
        except ValueError:
            assert (
                False
            ), f"calling inverse matrix square root on {spd} raised a value error"

    @pytest.mark.parametrize("snd", snd_matrices)
    def test_npd_inverse_sqrt_all_exceptions(self, snd):
        # Assert that exception is ALWAYS raised for seminegative definite matrices
        with pytest.raises(ValueError):
            radial_basis.inverse_matrix_sqrt(snd)

    @pytest.mark.parametrize("nonsym", nonsym_matrices)
    def test_nonsymmetric_inverse_sqrt_all_exceptions(self, nonsym):
        # Assert that exception is ALWAYS raised for nonsymmetric matrices
        with pytest.raises(ValueError):
            radial_basis.inverse_matrix_sqrt(nonsym)

    @pytest.mark.parametrize("spd", spd_matrices)
    def test_spd_inverse_sqrt(self, spd):
        dim = np.shape(spd)[0]
        inv_sqrt_s = radial_basis.inverse_matrix_sqrt(spd)
        assert_allclose(
            np.eye(dim), inv_sqrt_s @ inv_sqrt_s @ spd, rtol=1e-6, atol=1e-6
        )

    # @pytest.mark.parametrize("basis_size", basis_sizes)
    # def test_orthonormality(self, basis_size):
    #     # Create an array of GTO order n and GTO width sigma for testing orthonormality of basis set.
    #     # Set upper limit of n to be 20, sigma to be 3
    #     n_grid = np.random.randint(1, 4, basis_size)
    #     sigma_grid = np.random.uniform(0.1, 3, basis_size)
    #     # mesh to evaluate GTOs
    #     n_points = 100000
    #     r_mesh = np.linspace(0, 10, n_points)
    #     gto = np.zeros((basis_size, n_points))
    #     for i in range(basis_size):
    #         n = n_grid[i]
    #         sigma = sigma_grid[i]
    #         gto_n_sigma = r_mesh**n * np.exp(-(r_mesh**2) / (2 * sigma**2))
    #         gto[i, :] = gto_n_sigma
    #
    #     S = radial_basis.gto_overlap(
    #         n_grid[:, np.newaxis],
    #         n_grid[np.newaxis, :],
    #         sigma_grid[:, np.newaxis],
    #         sigma_grid[np.newaxis, :],
    #     )
    #     gto_orthonorm = radial_basis.inverse_matrix_sqrt(S) @ gto
    #
    #     # Evaluate the overlap between orthonormal GTOs using trapezoidal integration.
    #     # This is integration scheme is pretty numerically bad it's just used to prove a point.
    #     # Hence, we use a pretty large absolute tolerance.
    #     overlaps = np.zeros((basis_size, basis_size))
    #     for i in range(basis_size):
    #         for j in range(basis_size):
    #             overlap = np.trapz(
    #                 gto_orthonorm[i, :] * gto_orthonorm[j, :] * r_mesh**2, r_mesh
    #             )
    #             overlaps[i][j] = overlap
    #
    #     assert_allclose(np.eye(basis_size), overlaps, rtol=0, atol=1e-2)
