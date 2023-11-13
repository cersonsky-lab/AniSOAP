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

    def test_notimplemented_basis(self):
        with pytest.raises(ValueError):
            basis = RadialBasis(radial_basis="nonsense", max_angular=5, cutoff_radius=5)

    def test_radial_functions_n5(self):
        basis_gto = RadialBasis(radial_basis="monomial", max_angular=5, cutoff_radius=5)
        num_ns = basis_gto.get_num_radial_functions()

        # Compare against exact results
        num_ns_exact = [3, 3, 2, 2, 1, 1]
        assert len(num_ns) == len(num_ns_exact)
        for l, num in enumerate(num_ns):
            assert num == num_ns_exact[l]

    def test_radial_functions_n6(self):
        basis_gto = RadialBasis(radial_basis="monomial", max_angular=6, cutoff_radius=5)
        num_ns = basis_gto.get_num_radial_functions()

        # Compare against exact results
        num_ns_exact = [4, 3, 3, 2, 2, 1, 1]
        assert len(num_ns) == len(num_ns_exact)
        for l, num in enumerate(num_ns):
            assert num == num_ns_exact[l]

    def test_radial_functions_n7(self):
        basis_gto = RadialBasis(
            radial_basis="monomial", max_angular=6, max_radial=5, cutoff_radius=5
        )
        num_ns = basis_gto.get_num_radial_functions()

        # We specify max_radial so it's decoupled from max_angular.
        num_ns_exact = [5, 5, 5, 5, 5, 5, 5]
        assert len(num_ns) == len(num_ns_exact)
        for l, num in enumerate(num_ns):
            assert num == num_ns_exact[l]

    def test_radial_functions_n8(self):
        basis_gto = RadialBasis(
            radial_basis="monomial", max_angular=6, max_radial=[1, 2, 3, 4, 5, 6, 7], cutoff_radius=5
        )
        num_ns = basis_gto.get_num_radial_functions()

        # We specify max_radial so it's decoupled from max_angular.
        num_ns_exact = [1, 2, 3, 4, 5, 6, 7]
        assert len(num_ns) == len(num_ns_exact)
        for l, num in enumerate(num_ns):
            assert num == num_ns_exact[l]

class TestBadInputs:
    """
    Class for testing if radial_basis fails with bad inputs
    """
    DEFAULT_HYPERS = {
        "max_angular": 10,
        "radial_basis": "gto",
        "radial_gaussian_width": 5.0,
        "cutoff_radius": 1.0,
    }
    test_hypers = [
        # [
        #     {**DEFAULT_HYPERS, "radial_gaussian_width": 5.0, "max_radial": 3},
        #     ValueError,
        #     "Only one of max_radial or radial_gaussian_width can be independently specified",
        # ],
        [
            {**DEFAULT_HYPERS, "radial_gaussian_width": 5.0, "max_radial": [1, 2, 3]},  # default max_angular = 10
            ValueError,
            "If you specify a list of number of radial components, this list must be of length 11. Received 3."
        ],
        [
            {**DEFAULT_HYPERS, "radial_gaussian_width": 5.0, "max_radial": "nonsense"},
            ValueError,
            "`max_radial` must be None, int, or list of int"
        ],
        [
            {**DEFAULT_HYPERS, "radial_gaussian_width": 5.0, "max_radial": [1, "nonsense", 2]},
            ValueError,
            "`max_radial` must be None, int, or list of int"
        ],

    ]

    @pytest.mark.parametrize("hypers,error_type,expected_message", test_hypers)
    def test_hypers(self, hypers, error_type, expected_message):
        with pytest.raises(error_type) as cm:
            RadialBasis(**hypers)
            assert cm.message == expected_message

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
        basis_mon = RadialBasis(radial_basis="monomial", max_angular=2, cutoff_radius=5)
        basis_gto = RadialBasis(
            radial_basis="gto",
            radial_gaussian_width=sigma,
            max_angular=2,
            cutoff_radius=5,
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
            radial_basis="gto",
            radial_gaussian_width=sigma,
            max_angular=2,
            cutoff_radius=5,
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

    def test_nogto_warning(self):
        with pytest.warns(UserWarning):
            lmax = 5
            non_gto_basis = RadialBasis("monomial", lmax, cutoff_radius=5)
            # As a proxy for a tensor map, pass in a numpy array for features
            features = np.random.random((5, 5))
            non_normalized_features = non_gto_basis.orthonormalize_basis(features)
            assert_allclose(features, non_normalized_features)

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
