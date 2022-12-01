import numpy as np

def spherical_to_cartesian(lmax, nmaxs):
    assert len(nmaxs) == lmax
    T = []
    for l, nmax in enumerate(nmaxs):
        # Initialize array in which to store all
        # # coefficients for each l
        maxdeg = l + 2*nmax

        # Usage T_l[m,n,n0,n1,n2]
        T_l = np.zeros((2*l+1,nmax,maxdeg+1,maxdeg+1,maxdeg+1))
        
        # TODO: Compute coefficients

        T.append(T_l)

    return T