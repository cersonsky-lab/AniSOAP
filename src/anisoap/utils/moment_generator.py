import numpy as np
from numpy.testing import assert_allclose
from scipy.special import gamma
from math import comb

# Define function to compute all moments for a single
# variable Gaussian.
def compute_moments_single_variable(A, a, maxdeg):
    """
    Parameters:
    - A: float
        inverse of variance (see below for exact mathematical form)
    - a: float
        Center of Gaussian function
    - maxdeg: int
        Maximum degree for which the moments need to be computed.

    Returns:
    - A numpy array of size (maxdeg+1,) containing the moments defined as
        <x^n> = integral x^n exp(-A(x-a)^2/2) dx
        Note that the Gaussian is not normalized, meaning that we
        need to multiply all results by a global factor if we wish
        to interpret these as moments of a probability density.
    """
    assert maxdeg > 0

    # Initialize the values to start iterative computation
    moments = np.zeros((maxdeg + 1,))
    moments[0] = np.sqrt(2 * np.pi / A)
    moments[1] = a * moments[0]

    # Iteratively compute the next moments from the previous two
    for deg in range(1, maxdeg):
        moments[deg + 1] = a * moments[deg] + deg * moments[deg - 1] / A

    return moments


# Define function to compute all moments for a diagonal dilation matrix.
# The implementation focuses on conceptual simplicity, while sacrifizing
# memory efficiency.
# To be more specific, the array "moments" allows us to access the value
# of the moment <x^n0 * y^n1 * z^n2> simply as moments[n0,n1,n2].
# This leads to more intuitive code, at the cost of wasting around
# a third of the memory in the array to store zeros.
def compute_moments_diagonal_inefficient_implementation(
    principal_components, a, maxdeg
):
    """
    Parameters:
    - principal_components: np.ndarray of shape (3,)
        Array containing the three principal components
    - a: np.ndarray of shape (3,)
        Vectorial center of the trivariate Gaussian
    - maxdeg: int
        Maximum degree for which the moments need to be computed.

    Returns:
    - moments: np.ndarray of shape (3, maxdeg+1)
        moments[n0,n1,n2] is the (n0,n1,n2)-th moment of the Gaussian defined as

        .. math::
        <x^{n_0} * y^{n_1} * z^{n_2}> = \int(x^{n_0} * y^{n_1} * z^{n_2}) * \exp(-0.5*(r-a).T@cov@(r-a)) dxdydz
        \sum_{i=1}^{\\infty} x_{i}

        Note that the term "moments" in probability theory are defined for normalized Gaussian distributions.
        Here, we take the Gaussian without prefactor, meaning that all moments are scaled
        by a global factor.
    """
    # Initialize the array in which to store the moments
    # moments[n0, n1, n2] will be set to <x^n0 * y^n1 * z^n2>
    # This representation is very inefficient, since only about 1/6 of the
    # array elements will actually be relevant.
    # The advantage, however, is the simplicity in later use.
    moments = np.zeros((3, maxdeg + 1))

    # Precompute the single variable moments in x- y- and z-directions:
    moments_x = compute_moments_single_variable(principal_components[0], a[0], maxdeg)
    moments_y = compute_moments_single_variable(principal_components[1], a[1], maxdeg)
    moments_z = compute_moments_single_variable(principal_components[2], a[2], maxdeg)

    # Compute values for all relevant components for which the monomial degree is <= maxdeg
    for n0 in range(maxdeg + 1):
        for n1 in range(maxdeg + 1 - n0):
            for n2 in range(maxdeg + 1 - n0 - n1):
                deg = n0 + n1 + n2

                # Make sure that the degree is not above the maximal degree,
                # since we only need moments up to degree maxdeg.
                # (this is where we are wasting memory)
                if deg > maxdeg:
                    continue

                # For diagonal dilation matrices, the integral is a product
                # of factors that only depend on x, y and z, respectively.
                # Thus, the moment is the product of the x- y- and z-integrals.
                moments[n0, n1, n2] = moments_x[n0] * moments_y[n1] * moments_z[n2]

    return moments


# Define function to compute all moments for a general dilation matrix.
# The implementation focuses on conceptual simplicity, while sacrifizing
# memory efficiency.
# To be more specific, the array "moments" allows us to access the value
# of the moment <x^n0 * y^n1 * z^n2> simply as moments[n0,n1,n2].
# This leads to more intuitive code, at the cost of wasting around
# a third of the memory in the array to store zeros.
def compute_moments_inefficient_implementation(A, a, maxdeg):
    """
    Parameters:
    - A: symmetric 3x3 matrix (np.ndarray of shape (3,3))
        Dilation matrix of the Gaussian that determines its shape.
        It can be written as cov = RDR^T, where R is a rotation matrix that specifies
        the orientation of the three principal axes, while D is a diagonal matrix
        whose three diagonal elements are the lengths of the principal axes.
    - a: np.ndarray of shape (3,)
        Vectorial center of the trivariate Gaussian.
    - maxdeg: int
        Maximum degree for which the moments need to be computed.

    Returns:
    - The list of moments defined as

        .. math::
        <x^{n_0} * y^{n_1} * z^{n_2}> = \int(x^{n_0} * y^{n_1} * z^{n_2}) * \exp(-0.5*(r-a).T@cov@(r-a)) dxdydz
        \sum_{i=1}^{\\infty} x_{i}

        Note that the term "moments" in probability theory are defined for normalized Gaussian distributions.
        Here, we take the Gaussian
    """
    # Make sure that the provided arrays have the correct dimensions and properties
    assert A.shape == (3, 3), "Dilation matrix needs to be 3x3"
    assert np.sum((A - A.T) ** 2) < 1e-14, "Dilation matrix needs to be symmetric"
    assert a.shape == (3,), "Center of Gaussian has to be given by a 3-dim. vector"
    assert maxdeg > 0, "The maximum degree needs to be at least 1"
    cov = np.linalg.inv(A)  # the covariance matrix is the inverse of the matrix A
    global_factor = (2 * np.pi) ** 1.5 / np.sqrt(
        np.linalg.det(A)
    )  # normalization of Gaussian

    # Initialize the array in which to store the moments
    # moments[n0, n1, n2] will be set to <x^n0 * y^n1 * z^n2>
    # This representation is memory inefficient, since only about 1/3 of the
    # array elements will actually be relevant.
    # The advantage, however, is the simplicity in later use.
    moments = np.zeros((maxdeg + 1, maxdeg + 1, maxdeg + 1))

    # Initialize the first few elements
    moments[0, 0, 0] = 1.0
    moments[1, 0, 0] = a[0]  # <x>
    moments[0, 1, 0] = a[1]  # <y>
    moments[0, 0, 1] = a[2]  # <z>
    if maxdeg == 1:
        return global_factor * moments

    # Initialize the quadratic elements
    moments[2, 0, 0] = cov[0, 0] + a[0] ** 2
    moments[0, 2, 0] = cov[1, 1] + a[1] ** 2
    moments[0, 0, 2] = cov[2, 2] + a[2] ** 2
    moments[1, 1, 0] = cov[0, 1] + a[0] * a[1]
    moments[0, 1, 1] = cov[1, 2] + a[1] * a[2]
    moments[1, 0, 1] = cov[2, 0] + a[2] * a[0]
    if maxdeg == 2:
        return global_factor * moments

    # Iterate over all possible exponents to generate all moments
    # Instead of iterating over n1, n2 and n3, we iterate over the total degree of the monomials
    # which will allow us to simplify certain edge cases.

    for deg in range(2, maxdeg):
        for n0 in range(deg + 1):
            for n1 in range(deg + 1 - n0):
                # We consider monomials of degree "deg", and generate moments of degree deg+1.
                n2 = deg - n0 - n1

                # Run the x-iteration
                # Note that at this point, we are in some cases accessing
                # elements for which n1 = -1 or n2 = -1 due to the
                # subtraction. Mathematically, moments for which one of the
                # exponents is negative are set to zero.
                # With the current implementation, these are automatically
                # taken care of, since all array elements are initialized
                # to zero, and e.g. moments[3,-1,0] == moments[3, maxdeg-1, 0]
                # are properly treated as zero.
                moments[n0 + 1, n1, n2] = (
                    a[0] * moments[n0, n1, n2]
                    + cov[0, 0] * n0 * moments[n0 - 1, n1, n2]
                )

                if n1 > 0:
                    moments[n0 + 1, n1, n2] += cov[0, 1] * n1 * moments[n0, n1 - 1, n2]
                if n2 > 0:
                    moments[n0 + 1, n1, n2] += cov[0, 2] * n2 * moments[n0, n1, n2 - 1]

                # If n0 is equal to zero, we also need the y- and z-iterations
                if n0 == 0:
                    # Run the y-iteration
                    moments[n0, n1 + 1, n2] = a[1] * moments[n0, n1, n2]
                    if n1 > 0:
                        moments[n0, n1 + 1, n2] += (
                            cov[1, 1] * n1 * moments[n0, n1 - 1, n2]
                        )
                    if n2 > 0:
                        moments[n0, n1 + 1, n2] += (
                            cov[1, 2] * n2 * moments[n0, n1, n2 - 1]
                        )

                    if n0 == 0 and n1 == 0:
                        # Run the z-iteration
                        moments[n0, n1, n2 + 1] = (
                            a[2] * moments[n0, n1, n2]
                            + cov[2, 2] * n2 * moments[n0, n1, n2 - 1]
                        )

    return global_factor * moments
