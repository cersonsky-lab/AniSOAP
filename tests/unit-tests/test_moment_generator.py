import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import (
    comb,
    gamma,
)

# Import the different versions of the moment generators
from anisoap.utils import (
    assert_close,
    compute_moments_diagonal_inefficient_implementation,
    compute_moments_inefficient_implementation,
    compute_moments_single_variable,
)


class TestMomentsUnivariateGaussian:
    """
    Class for testing the correctness of the single variable moments
    <x^n> = integral x^n exp(-0.5*A*x^2)
    against analytical expressions or alternative evaluation schemes.
    """

    maxdegs = [2, 5, 10]
    sigmas = np.geomspace(0.2, 5, 7)
    centers = np.linspace(-10, 10, 7)

    # Test the single variable moments against the analytical expression
    # for the special case of Gaussians centered around zero.
    @pytest.mark.parametrize("maxdeg", maxdegs)
    @pytest.mark.parametrize("sigma", sigmas)
    def test_moments_single_variable_centered(self, sigma, maxdeg):
        A = 1 / sigma**2
        moments = compute_moments_single_variable(A, 0, maxdeg)

        moments_exact = np.zeros((maxdeg + 1,))
        for deg in range(maxdeg + 1):
            exact_value = 0
            if deg % 2 == 0:
                neff = (deg + 1) / 2
                exact_value = (2 * sigma**2) ** neff * gamma(neff)
            moments_exact[deg] = exact_value

        assert_allclose(moments, moments_exact, rtol=1e-15, atol=1e-15)

    # Test the single variable moments against the analytical expression
    # for the general case of a Gaussian centered around an arbitrary
    # point a.
    # The reference values are obtained from an alternative approach
    # to compute the moments using the binomial theorem.
    @pytest.mark.parametrize("maxdeg", maxdegs)
    @pytest.mark.parametrize("center", centers)
    @pytest.mark.parametrize("sigma", sigmas)
    def test_moments_single_variable_non_centered(self, sigma, center, maxdeg):
        A = 1 / sigma**2

        # Compute the exact moments for the centered moments
        centered_moments = np.zeros((maxdeg + 1,))
        for deg in range(maxdeg + 1):
            exact_value = 0
            if deg % 2 == 0:
                neff = (deg + 1) / 2
                exact_value = (2 / A) ** neff * gamma(neff)
            centered_moments[deg] = exact_value

        # Compute the moments from the binomial theorem
        moments = np.zeros((maxdeg + 1,))
        for deg in range(maxdeg + 1):
            # The first term in the binomial expansion
            # is always the centered moment of the same degree
            moments[deg] += centered_moments[deg]

            # Get the correction from the centered moment
            for k in range(deg):
                moments[deg] -= comb(deg, k) * (-center) ** (deg - k) * moments[k]

        # Compare with the moments obtained from the iterative algorithm
        # used in the main code
        moments_from_code = compute_moments_single_variable(A, center, maxdeg)
        assert_allclose(moments_from_code, moments, rtol=1e-12, atol=1e-15)


# Function that returns all moments <x^n0 * y^n1 * z^n2>
# for the trivariate Gaussian up to degree 3, i.e. n0+n1+n2 <=3,
# using the exact analytical formulae.
# It is used to compare the moments obtained from
# the two iterative algorithms, i.e. the general algorithm
# and the special case that only works for diagonal dilation matrices.
def get_exact_moments(A, a, maxdeg=3):
    cov = np.linalg.inv(A)
    global_factor = (2 * np.pi) ** 1.5 / np.sqrt(np.linalg.det(A))
    assert maxdeg in [1, 2, 3]

    moments_exact = np.zeros((maxdeg + 1, maxdeg + 1, maxdeg + 1))
    moments_exact[0, 0, 0] = 1.0
    # Exact expressions for degree 1
    moments_exact[1, 0, 0] = a[0]
    moments_exact[0, 1, 0] = a[1]
    moments_exact[0, 0, 1] = a[2]
    if maxdeg == 1:
        return global_factor * moments_exact

    # Exact expressions for degree 2
    moments_exact[2, 0, 0] = cov[0, 0] + a[0] ** 2
    moments_exact[0, 2, 0] = cov[1, 1] + a[1] ** 2
    moments_exact[0, 0, 2] = cov[2, 2] + a[2] ** 2
    moments_exact[1, 1, 0] = cov[0, 1] + a[0] * a[1]
    moments_exact[0, 1, 1] = cov[1, 2] + a[1] * a[2]
    moments_exact[1, 0, 1] = cov[0, 2] + a[0] * a[2]
    if maxdeg == 2:
        return global_factor * moments_exact

    # Exact expressions for degree 3
    moments_exact[3, 0, 0] = 3 * a[0] * cov[0, 0] + a[0] ** 3
    moments_exact[0, 3, 0] = 3 * a[1] * cov[1, 1] + a[1] ** 3
    moments_exact[0, 0, 3] = 3 * a[2] * cov[2, 2] + a[2] ** 3
    moments_exact[2, 1, 0] = a[1] * (cov[0, 0] + a[0] ** 2) + 2 * a[0] * cov[0, 1]
    moments_exact[2, 0, 1] = a[2] * (cov[0, 0] + a[0] ** 2) + 2 * a[0] * cov[0, 2]
    moments_exact[1, 2, 0] = a[0] * (cov[1, 1] + a[1] ** 2) + 2 * a[1] * cov[1, 0]
    moments_exact[0, 2, 1] = a[2] * (cov[1, 1] + a[1] ** 2) + 2 * a[1] * cov[1, 2]
    moments_exact[1, 0, 2] = a[0] * (cov[2, 2] + a[2] ** 2) + 2 * a[2] * cov[2, 0]
    moments_exact[0, 1, 2] = a[1] * (cov[2, 2] + a[2] ** 2) + 2 * a[2] * cov[2, 1]
    moments_exact[1, 1, 1] = (
        a[0] * a[1] * a[2] + a[0] * cov[1, 2] + a[1] * cov[0, 2] + a[2] * cov[0, 1]
    )
    if maxdeg == 3:
        return global_factor * moments_exact


class TestMomentsTrivariateGaussian:
    """
    Class for testing the values of the moments for a trivariate Gaussian
    defined as
    <x^n0 * y^n1 * z^n2> = integral (x^n0 * y^n1 * z^n2) * exp(-0.5*(r-a).T@cov@(r-a)) dxdydz.
    """

    principal_components_list = [[1.3, 2.3, 5.1], [0.2, 2.1, 3.9]]
    maxdegs_small = [1, 2, 3]
    maxdegs_any = [3, 6, 10]
    centers = [np.array([2.3, -3.1, 6.2]), np.array([-2.1, -5.5, -0.43])]

    # Test the diagonal implementation against the analytical results
    # for degrees up to 3.
    @pytest.mark.parametrize("maxdeg", maxdegs_small)
    @pytest.mark.parametrize("center", centers)
    @pytest.mark.parametrize("principal_components", principal_components_list)
    def test_moments_diagonal_vs_exact(self, principal_components, center, maxdeg):
        A = np.diag(principal_components)
        moments_diagonal = compute_moments_diagonal_inefficient_implementation(
            principal_components, center, maxdeg
        )
        moments_exact = get_exact_moments(A, center, maxdeg)

    # For isotropic Gaussians, many moments will be the same due to symmetry.
    # e.g. <x^2y> = <x^2z> = <xy^2> = <xz^2> = <y^2z> = <yz^2>
    # We test that such coefficient groups do indeed have the same moment.
    centers_1d = np.linspace(-5, 5, 3)
    sigmas = np.geomspace(0.2, 5, 7)

    @pytest.mark.parametrize("maxdeg", maxdegs_any)
    @pytest.mark.parametrize("center_offset", centers_1d)
    @pytest.mark.parametrize("sigma", sigmas)
    def test_moments_diagonal_permutation_symmetry(self, sigma, center_offset, maxdeg):
        center = center_offset * np.ones((3,))
        principal_components = np.ones((3,)) / sigma**2
        moments_diagonal = compute_moments_diagonal_inefficient_implementation(
            principal_components, center, maxdeg
        )
        eps = 1e-12
        for n0 in range(maxdeg + 1):
            for n1 in range(maxdeg + 1):
                for n2 in range(maxdeg + 1):
                    deg = n0 + n1 + n2
                    if deg > maxdeg:
                        assert moments_diagonal[n0, n1, n2] == 0
                    else:
                        assert_close(
                            moments_diagonal[n0, n1, n2], moments_diagonal[n1, n0, n2]
                        )
                        assert_close(
                            moments_diagonal[n0, n1, n2], moments_diagonal[n2, n1, n0]
                        )
                        assert_close(
                            moments_diagonal[n0, n1, n2], moments_diagonal[n0, n2, n1]
                        )

    # For diagonal dilation matrices, the general algorithm to generate
    # the moments should lead to the same results as the specialized
    # implementation for diagonal matrices.
    @pytest.mark.parametrize("maxdeg", maxdegs_any)
    @pytest.mark.parametrize("center", centers)
    @pytest.mark.parametrize("principal_components", principal_components_list)
    def test_moments_diagonal_vs_general(self, principal_components, center, maxdeg):
        A = np.diag(principal_components)
        moments_general = compute_moments_inefficient_implementation(A, center, maxdeg)
        moments_diagonal = compute_moments_diagonal_inefficient_implementation(
            principal_components, center, maxdeg
        )
        assert_allclose(moments_general, moments_diagonal, rtol=1e-15, atol=3e-16)

    # Test the moments obtained from the general code against the exact analytical expression.
    # Generate two arbitrary positive definitive symmetric matrices
    As = [np.array([[1.0, 3, 5.0], [3.0, 7.0, 9.0], [5.0, 9.0, 13.0]]) + 8.1]
    As.append(
        np.diag([2.1, 3.2, 4]) + 1e-1 * np.array([[1, 2, 3], [2, 1, 4], [3, 4, 2.1]])
    )

    @pytest.mark.parametrize("maxdeg", maxdegs_small)
    @pytest.mark.parametrize("center", centers)
    @pytest.mark.parametrize("A", As)
    def test_moments_general_vs_exact(self, A, center, maxdeg):
        moments_general = compute_moments_inefficient_implementation(A, center, maxdeg)
        moments_exact = get_exact_moments(A, center, maxdeg)
        assert_allclose(moments_general, moments_exact)
