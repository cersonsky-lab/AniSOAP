import numpy as np
import scipy
from scipy.special import factorial, factorial2
from math import comb
from anisoap.utils import monomial_iterator

def get_P_ll(l,x):
    return factorial2(2*l-1)*(-1)**l* np.sqrt(1-x**2)**l

#We are implementing iterations of the form R_{l+1} = prefact_minus0* z * R_{l} + prefact_minus1* r2 * R_{l-1}
def prefact_minus0(l):
    m=np.arange(-l,l+1)
    return np.sqrt(factorial(l+1-m)/factorial(l+1+m)) * np.sqrt(factorial(l+m)/factorial(l-m)) * (2*l+1)/(l+1-m)
    
def prefact_minus1(l):
    m=np.arange(-l+1,l)
    return -1* np.sqrt(factorial(l+1-m)/factorial(l+1+m)) * np.sqrt(factorial(l-1+m)/factorial(l-1-m)) * (l+m)/(l+1-m)

def binom(n, k):
    return comb(n, k)
#     return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def spherical_to_cartesian(lmax, num_ns):
    assert len(num_ns) == lmax + 1

    # Initialize array in which to store all
    # coefficients for each l
    # Usage T_l[m,n,n0,n1,n2]
    T = []
    for l, num_n in enumerate(num_ns):
        maxdeg = l + 2*(num_n-1)
        T_l = np.zeros((2*l+1,num_n,maxdeg+1,maxdeg+1,maxdeg+1))
        T.append(T_l)

    # Initialize array in which to store all coefficients for each l
    # Usage T_l[m,n,n0,n1,n2]
    T[0][0,0,0,0,0] = 1
    for l in range(1,lmax+1):
        prefact = np.sqrt(2) * factorial2(2*l-1) / np.sqrt(factorial(2*l))
        for k in range(l//2+1):
            n1 = 2*k
            n0 = l-n1
            T[l][2*l,0, n0, n1,0] = binom(l, n1) *(-1)**k  
        for k in range((l-1)//2+1):
            n1 = 2*k+1
            n0 = l-n1
            T[l][0,0,n0,n1,0] = binom(l, n1) *(-1)**k
        T[l]*= prefact

    # Run iteration over (l,m) to generate all coefficients for n=0.
    for l in range(lmax):
        deg = l
        myiter = iter(monomial_iterator.TrivariateMonomialIndices(deg+1))
        for idx,n0,n1,n2 in myiter:
            a = prefact_minus0(l) # length 2l+1 due to m dependence
            b = prefact_minus1(l) # length 2l+1 due to m dependence
    
            #(-l+1)+2: (l+1) -2 gets contributions from T[l-1]
            if n0-2>=0:
                T[l+1][2:2*l+1,0,n0,n1,n2] += b * T[l-1][:,0,n0-2,n1,n2]
            if n1-2>=0:
                T[l+1][2:2*l+1,0,n0,n1,n2] += b * T[l-1][:,0,n0,n1-2,n2]
            if n2-2>=0:
                T[l+1][2:2*l+1,0,n0,n1,n2] += b * T[l-1][:,0,n0,n1,n2-2]
            #(-l+1)+1: (l+1) -1 gets contributions from T[l]
            if n2-1>=0:
                T[l+1][1:2*l+2,0,n0,n1,n2] += a * T[l][:,0,n0,n1,n2-1]

    # Run the iteration over n
    for l in range(lmax+1):
        for n in range(1,num_ns[l]):
            deg = l + 2*n # degree of polynomial
            myiter = iter(monomial_iterator.TrivariateMonomialIndices(deg))
            for idx,n0,n1,n2 in myiter:
                # Use recurrence relation to update
                # Warning, if n0-2, n1-2 or n2-2 are negative
                # it might be necessary to add if statements
                # to avoid.
                if n0>=2:
                    T[l][:,n,n0,n1,n2] += T[l][:,n-1,n0-2,n1,n2]
                if n1>=2:
                    T[l][:,n,n0,n1,n2] += T[l][:,n-1,n0,n1-2,n2]
                if n2>=2:
                    T[l][:,n,n0,n1,n2] += T[l][:,n-1,n0,n1,n2-2]

    return T
