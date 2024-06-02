import numpy as np
import scipy
from scipy.special import (
    comb,
    factorial,
    factorial2,
)

from anisoap.utils import monomial_iterator

# Here we are implementing recurrence of the form R_{l}^m = prefact_minus1* z * T_{l-1} + prefact_minus2* r2 * T_{l-2}
# where R_l^m is a solid harmonic, when expressed on a monomial basis - R_l^m = \sum_{n0+n1+n2=l} T_{l}[n0,n1,n2] x^n0 y^n1 z^n2
# We will further define these coefficients with an additional n-dependent prefactor r^P2n}
# r^2n R_l^m = \sum_{n0,n1,n2| (n0+n1+n2=l+2n)} T_{nlm}[n0,n1,n2] x^n0 y^n1 z^n2


def prefact_minus1(l):
    r"""Computes the prefactor that multiplies the :math:`T_{l-1}^\text{th}` term in the iteration.

    For :math:`m \in \left[-l, -l+2, ..., l \right]`, compute the factor as

    .. math::

        \left( \frac{\sqrt{(l+1-m)!}}{(l+1+m)!} \right) \left( \frac{\sqrt{(l+m)!}}{(l-m)!} \right)
            \left( \frac{2l+1}{l+1-m} \right)

    Parameters
    ----------
    l : int
        Term immediately proceeding the term for which the prefactor is computed.

    Returns
    -------
    list of size (2l + 1)
        corresponds to the prefactor that multiplies the :math:`T_{l-1}^\text{th}`
        term in the iteration

    """
    m = np.arange(-l, l + 1)
    return (
        np.sqrt(factorial(l + 1 - m) / factorial(l + 1 + m))
        * np.sqrt(factorial(l + m) / factorial(l - m))
        * (2 * l + 1)
        / (l + 1 - m)
    )


def prefact_minus2(l):
    r"""Computes the prefactor that multiplies the :math:`T_{l-2}^\text{th}` term in the iteration.

    For :math:`m \in \left[-l+1, -l+2, ..., l-1\right]`, compute the factor as

    .. math::

        \left( \frac{\sqrt{(l+1-m)!}}{(l+1+m)!} \right) \left(\frac{\sqrt{(l-1+m)!}}{(l-1-m)!} \right)
            \left( \frac{l+m}{l-m+1} \right)

    Parameters
    ----------
    l : int
        Term two places after the term for which the prefactor is computed

    Returns
    -------
    list of size (2l - 1)
        Corresponds to the prefactor that multiplies the term in question

    """
    m = np.arange(-l + 1, l)
    return (
        -1
        * np.sqrt(factorial(l + 1 - m) / factorial(l + 1 + m))
        * np.sqrt(factorial(l - 1 + m) / factorial(l - 1 - m))
        * (l + m)
        / (l + 1 - m)
    )


def binom(n, k):
    return comb(n, k)


#     return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def spherical_to_cartesian(lmax, num_ns):
    """
    Finds the coefficients for the cartesian polynomial form of solid harmonics 
    :math:`R_{lm} = sqrt((4pi)/(2l+1))*r^l*Y_{lm}`.  Note that our AniSOAP 
    expansion does not contain the sqrt((4pi)/(2l+1)), so in calculating 
    expansion coefficients, we need to divide by that coefficient.

    Parameters
    ----------
    lmax : int

    num_ns : int

    """
    assert len(num_ns) == lmax + 1

    # Initialize array in which to store all
    # coefficients for each l, stored as  T_l[m,n,n0,n1,n2] where n0,n1,n2 correspond to the respective powers in x^n0 y^n1 z^n2
    T = []
    for l, num_n in enumerate(num_ns):
        maxdeg = l + 2 * (num_n - 1)  # maxdeg = 2*l +n
        T_l = np.zeros((2 * l + 1, num_n, maxdeg + 1, maxdeg + 1, maxdeg + 1))
        T.append(T_l)

    # Initial conditions of the recurrence T[l][m=l,n=0, n0, l-n0, 0] and T[l][m=-l, n=0, n0, l-n0,0]
    T[0][0, 0, 0, 0, 0] = 1
    for l in range(1, lmax + 1):
        prefact = np.sqrt(2) * factorial2(2 * l - 1) / np.sqrt(factorial(2 * l))
        for k in range(l // 2 + 1):
            n1 = 2 * k
            n0 = l - n1
            T[l][2 * l, 0, n0, n1, 0] = binom(l, n1) * (-1) ** k
        for k in range((l - 1) // 2 + 1):
            n1 = 2 * k + 1
            n0 = l - n1
            T[l][0, 0, n0, n1, 0] = binom(l, n1) * (-1) ** k
        T[l] *= prefact

    # Run iteration over (l,m) to generate all coefficients for n=0.
    # T[l][:,0,n0,n1,n2] += prefact_minus1 * T[l-1][:,0,n0,n1,n2-1]
    # T[l][:,0,n0,n1,n2] += prefact_minus2 * T[l-2][:,0,n0-2,n1,n2]
    # T[l][:,0,n0,n1,n2] += prefact_minus2 * T[l-2][:,0,n0,n1-2,n2]
    # T[l][:,0,n0,n1,n2] += prefact_minus2 * T[l-2][:,0,n0,n1,n2-2]
    for l in range(lmax + 1):
        deg = l
        myiter = iter(monomial_iterator.TrivariateMonomialIndices(deg))
        for idx, n0, n1, n2 in myiter:
            a = prefact_minus1(l - 1)  # elements corresponding to m in (-l+2, ... l-2)
            b = prefact_minus2(l - 1)  # elements corresponding to m in (-l+1, .... l-1)

            # T[l][-(l-2).... (l-2)]  gets contributions from T[l-2][:]
            if n0 - 2 >= 0:
                T[l][2 : 2 * l - 1, 0, n0, n1, n2] += b * T[l - 2][:, 0, n0 - 2, n1, n2]
            if n1 - 2 >= 0:
                T[l][2 : 2 * l - 1, 0, n0, n1, n2] += b * T[l - 2][:, 0, n0, n1 - 2, n2]
            if n2 - 2 >= 0:
                T[l][2 : 2 * l - 1, 0, n0, n1, n2] += b * T[l - 2][:, 0, n0, n1, n2 - 2]
            # T[l][(-l-1)... (l-1)]  gets contributions from T[l-1]
            if n2 - 1 >= 0:
                T[l][1 : 2 * l, 0, n0, n1, n2] += a * T[l - 1][:, 0, n0, n1, n2 - 1]

    # Run the iteration over n
    # T[l][:,n,n0,n1,n2] += T[l][:,n-1,n0-2,n1,n2]
    # T[l][:,n,n0,n1,n2] += T[l][:,n-1,n0,n1-2,n2]
    # T[l][:,n,n0,n1,n2] += T[l][:,n-1,n0,n1,n2-2]
    for l in range(lmax + 1):
        for n in range(1, num_ns[l]):
            deg = l + 2 * n  # degree of polynomial
            myiter = iter(monomial_iterator.TrivariateMonomialIndices(deg))
            for idx, n0, n1, n2 in myiter:
                # Use recurrence relation to update
                if n0 >= 2:
                    T[l][:, n, n0, n1, n2] += T[l][:, n - 1, n0 - 2, n1, n2]
                if n1 >= 2:
                    T[l][:, n, n0, n1, n2] += T[l][:, n - 1, n0, n1 - 2, n2]
                if n2 >= 2:
                    T[l][:, n, n0, n1, n2] += T[l][:, n - 1, n0, n1, n2 - 2]

    return T
