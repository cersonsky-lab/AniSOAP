from re import I
import numpy as np
import pytest
from scipy.special import gamma
from math import comb
from numpy.testing import assert_allclose

from src.utils.moment_generator import compute_moments_single_variable, compute_moments_diagonal_inefficient_implementation, compute_moments_inefficient_implementation


def test_moments_single_variable_centered():
    sigma = 0.32
    a = 0.
    maxdeg = 5
    A = 1/sigma**2
    moments = compute_moments_single_variable(A, a, maxdeg)

    moments_exact = np.zeros((maxdeg+1,))
    for deg in range(maxdeg+1):
        exact_value = 0
        if deg % 2 == 0:
            neff = (deg + 1) / 2
            exact_value = (2 * sigma**2)**neff * gamma(neff)
        moments_exact[deg] = exact_value
    
    assert_allclose(moments, moments_exact)

def test_moments_single_variable_non_centered(A, a, maxdeg):
    # Compute the exact moments for the centered moments
    centered_moments = np.zeros((maxdeg+1,))
    for deg in range(maxdeg+1):
        exact_value = 0
        if deg % 2 == 0:
            neff = (deg + 1) / 2
            exact_value = (2 / A)**neff * gamma(neff)
        centered_moments[deg] = exact_value
    
    # Compute the moments from the binomial theorem
    moments = np.zeros((maxdeg+1,))
    for deg in range(maxdeg+1):
        # The first term in the binomial expansion
        # is always the centered moment of the same degree
        moments[deg] += centered_moments[deg]
        
        # Get the correction from the centered moment
        for k in range(deg):
            moments[deg] -= comb(deg, k) * (-a)**(deg-k) * moments[k]

    # Compare with the moments obtained from the iterative algorithm
    # used in the main code
    moments_from_code = compute_moments_single_variable(A, a, maxdeg)
    assert_allclose(moments_from_code, moments, rtol=1e-15, atol=1e-15)

class TestMomentsTrivariateGaussian():
    """
    Class for testing the values of the moments for the full
    trivariate Gaussian.
    """

    def test_moments_diagonal_vs_exact(principal_components, a, maxdeg=3):
        A = np.diag(principal_components)
        moments_diagonal = compute_moments_diagonal_inefficient_implementation(principal_components, a, maxdeg)
        moments_exact = get_exact_moments(A, a, maxdeg)

    # For isotropic Gaussians, many moments will be the same due to symmetry.
    # e.g. <x^2y> = <x^2z> = <xy^2> = <xz^2> = <y^2z> = <yz^2>
    def test_moments_diagonal_permutation_symmetry(principal_components, a, maxdeg):
        moments_diagonal = compute_moments_diagonal_inefficient_implementation(principal_components, a, maxdeg)
        eps = 1e-12
        for n0 in range(maxdeg+1):
            for n1 in range(maxdeg+1):
                for n2 in range(maxdeg+1):
                    deg = n0 + n1 + n2
                    if deg > maxdeg:
                        assert moments_diagonal[n0, n1, n2] == 0
                    else:
                        assert abs(moments_diagonal[n0,n1,n2]-moments_diagonal[n1,n0,n2]) < eps
                        assert abs(moments_diagonal[n0,n1,n2]-moments_diagonal[n2,n1,n0]) < eps
                        assert abs(moments_diagonal[n0,n1,n2]-moments_diagonal[n0,n2,n1]) < eps
    
    # For diagonal dilation matrices, the general algorithm to generate
    # the moments should lead to the same results as the specialized
    # implementation for diagonal matrices.
    def test_moments_diagonal_vs_general(principal_components, a, maxdeg):
        principal_components = np.array([2.8,0.4,1.1])
        A = np.diag(principal_components)
        a = np.array([3.1, -2.3, 5.92])
        maxdeg = 3
        moments_general = compute_moments_inefficient_implementation(A, a, maxdeg)
        moments_diagonal = compute_moments_diagonal_inefficient_implementation(principal_components, a, maxdeg)
        assert_allclose(moments_general, moments_diagonal, rtol=1e-15, atol=3e-16)
    
    # Test the general results against the exact analytical expression.
    def test_moments_general_vs_exact(A, a, maxdeg):
        moments_general = compute_moments_inefficient_implementation(A, a, maxdeg)
        moments_exact = get_exact_moments(A, a, maxdeg)
        assert_allclose(moments_general, moments_exact)


# Function that returns all moments up to degree 3
# using the exact analytical formulae.
# It is used to compare the moments obtained from
# the iterative algorithms.
def get_exact_moments(A, a, maxdeg=3):
    cov = np.linalg.inv(A)
    global_factor = (2*np.pi)**1.5 / np.sqrt(np.linalg.det(A))
    assert maxdeg in [1,2,3]
    
    moments_exact = np.zeros((maxdeg+1, maxdeg+1, maxdeg+1))
    moments_exact[0,0,0] = 1.
    # Exact expressions for degree 1
    moments_exact[1,0,0] = a[0]
    moments_exact[0,1,0] = a[1]
    moments_exact[0,0,1] = a[2]
    if maxdeg == 1:
        return global_factor * moments_exact

    # Exact expressions for degree 2
    moments_exact[2,0,0] = cov[0,0] + a[0]**2
    moments_exact[0,2,0] = cov[1,1] + a[1]**2
    moments_exact[0,0,2] = cov[2,2] + a[2]**2
    moments_exact[1,1,0] = a[0]*a[1]
    moments_exact[0,1,1] = a[1]*a[2]
    moments_exact[1,0,1] = a[0]*a[2]
    if maxdeg == 2:
        return global_factor * moments_exact

    # Exact expressions for degree 3
    moments_exact[3,0,0] = 3*a[0]*cov[0,0] + a[0]**3
    moments_exact[0,3,0] = 3*a[1]*cov[1,1] + a[1]**3
    moments_exact[0,0,3] = 3*a[2]*cov[2,2] + a[2]**3
    moments_exact[2,1,0] = a[1]*(cov[0,0] + a[0]**2) +  2*a[0]*cov[0,1]
    moments_exact[2,0,1] = a[2]*(cov[0,0] + a[0]**2) +  2*a[0]*cov[0,2]
    moments_exact[1,2,0] = a[0]*(cov[1,1] + a[1]**2) +  2*a[1]*cov[1,0]
    moments_exact[0,2,1] = a[2]*(cov[1,1] + a[1]**2) +  2*a[1]*cov[1,2]
    moments_exact[1,0,2] = a[0]*(cov[2,2] + a[2]**2) +  2*a[2]*cov[2,0]
    moments_exact[0,1,2] = a[1]*(cov[2,2] + a[2]**2) +  2*a[2]*cov[2,1]
    moments_exact[1,1,1] = a[0]*a[1]*a[2] + a[0]*cov[1,2] + a[1]*cov[0,2] + a[2]*cov[0,1]
    if maxdeg == 3:
        return global_factor * moments_exact


