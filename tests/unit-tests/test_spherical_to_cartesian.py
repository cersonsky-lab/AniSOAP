from math import factorial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from anisoap.utils import (
    TrivariateMonomialIndices,
    assert_close,
    spherical_to_cartesian,
)


# Generate the exact values of the spherical to
# Cartesian coordinate transformations for l=0,1,2.
# For l=0 and l=1, arbitrary degrees in n are supported,
# while for l=2, only n=0 is supported.
def spherical_to_cartesian_exact(lmax, num_ns):
    T = []
    assert lmax <= 3, "Exact analytical expressions only implemented for lmax<=3"
    assert (
        len(num_ns) == lmax + 1
    ), "Provided list of nmax values has to match number of l values"

    # Initialize the arrays in the same way as the main code
    for l, nmax in enumerate(num_ns):
        # Initialize array in which to store all
        # # coefficients for each l
        maxdeg = l + 2 * (nmax - 1)

        # Usage T_l[m,n,n0,n1,n2]
        shape = (2 * l + 1, nmax, maxdeg + 1, maxdeg + 1, maxdeg + 1)
        T_l = np.zeros((2 * l + 1, nmax, maxdeg + 1, maxdeg + 1, maxdeg + 1))
        T.append(T_l)

    # Fill in the exact coefficients for l=0
    for n in range(num_ns[0]):
        monomial_indices = iter(TrivariateMonomialIndices(deg=n))
        fac_n = factorial(n)
        for idx, n0, n1, n2 in monomial_indices:
            coeff = fac_n / factorial(n0) / factorial(n1) / factorial(n2)
            T[0][0, n, 2 * n0, 2 * n1, 2 * n2] = coeff
    if lmax == 0:
        return T

    # Fill in the exact coefficients for l=1
    for n in range(num_ns[1]):
        monomial_indices = iter(TrivariateMonomialIndices(deg=n))
        fac_n = factorial(n)
        for idx, n0, n1, n2 in monomial_indices:
            coeff = fac_n / factorial(n0) / factorial(n1) / factorial(n2)
            T[1][
                0, n, 2 * n0, 2 * n1 + 1, 2 * n2
            ] = coeff  # m=-1 with solid harmonic = y
            T[1][
                1, n, 2 * n0, 2 * n1, 2 * n2 + 1
            ] = coeff  # m=0 with solid harmonic = z
            T[1][
                2, n, 2 * n0 + 1, 2 * n1, 2 * n2
            ] = coeff  # m=1 with solid harmonic = x
    if lmax == 1:
        return T

    # Fill in the exact coefficients for l=2 and n=0.
    assert num_ns[2] == 1, "For l=2, only n=0 is implemented"
    T[2][0, 0, 1, 1, 0] = np.sqrt(3)  # m=-2
    T[2][1, 0, 0, 1, 1] = np.sqrt(3)  # m=-1
    T[2][3, 0, 1, 0, 1] = np.sqrt(3)  # m=+1

    # Coeffs for m=0
    T[2][2, 0, 2, 0, 0] = -0.5
    T[2][2, 0, 0, 2, 0] = -0.5
    T[2][2, 0, 0, 0, 2] = 1

    # Coeffs for m=+2
    T[2][4, 0, 2, 0, 0] = np.sqrt(3) / 2
    T[2][4, 0, 0, 2, 0] = -np.sqrt(3) / 2

    return T


@pytest.mark.parametrize("lmax", np.arange(5))
@pytest.mark.parametrize("num_n", np.arange(1, 5))
def test_compare_against_exact(lmax, num_n):
    assert num_n >= 1, "Number of radial channels has to be at least 1"

    # Compute the spherical to Cartesian transformation coefficients
    # using the calculator in the main code as well as the reference
    # code using analytical expressions that was defined above.
    T_ref = spherical_to_cartesian_exact(
        lmax=2, num_ns=[num_n, num_n, 1]
    )  # reference values
    T = spherical_to_cartesian(lmax, num_ns=[num_n] * (lmax + 1))

    # Compare the obtained coefficients against the exact values
    assert_allclose(T[0], T_ref[0], rtol=1e-14)
    if lmax >= 1:
        assert_allclose(T[1], T_ref[1], rtol=1e-14)
    if lmax >= 2:
        assert_allclose(T[2][:, 0, :3, :3, :3], T_ref[2][:, 0], rtol=1e-14)


class TestSumOfCoefficients:
    """
    While it is not as simple to generate all transformation coefficients
    from scratch using an alternative method, it is much simpler to check
    that the sum of all coefficients is equal to the correct value.
    This is being tested in this class.
    """

    lmaxs = np.arange(8)
    num_ns = np.arange(1, 14)

    # For l=0 and arbitrary n, the sum of all coefficients
    # has to equal 3^n.
    # This follows from the fact that the solid harmonics at
    # (l,m)=(0,0) is just equal to the constant 1 function,
    # and thus the coefficients are simply obtained from
    # the expansion of (x^2+y^2+z^2)^n.
    # The sum of all such coefficients is simply equal to
    # the special case of the above for x=y=z=1, meaning
    # that the sum of the coefficients will be equal to 3^n.
    @pytest.mark.parametrize("num_n", num_ns)
    def test_sum_for_l0(self, num_n):
        assert num_n >= 1
        lmax = 0
        T = spherical_to_cartesian(lmax, num_ns=[num_n])
        for n in range(num_n):
            T_0n = T[0][0, n]
            sum_of_all = np.sum(T_0n)
            assert_close(sum_of_all, 3.0**n, rtol=1e-14)

    # We now repeat the same test also for l=1,2,3.
    # The sum of coefficients for some n is always 3^n times
    # the sum of coefficients at n=0 which has been computed
    # explicity.
    @pytest.mark.parametrize("lmax", lmaxs)
    @pytest.mark.parametrize("num_n", num_ns)
    def test_sum_of_coefficients(self, lmax, num_n):
        assert num_n >= 1
        T = spherical_to_cartesian(lmax, num_ns=[1] * (lmax) + [num_n])
        ref_value = np.sum(T[lmax][:, 0])
        for n in range(num_n):
            sum_of_all = np.sum(T[lmax][:, n])
            assert_close(sum_of_all, ref_value * (3.0**n), rtol=1e-14)
