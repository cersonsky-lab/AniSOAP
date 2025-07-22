import metatensor
import numpy as np
from anisoap.representations import EllipsoidalDensityProjection
from anisoap.utils import ClebschGordanReal, cg_combine, standardize_keys
from ase.io import read
from matplotlib import pyplot as plt
from matplotlib import rc
from featomic import SoapPowerSpectrum
from sklearn.decomposition import PCA
from skmatter.metrics import global_reconstruction_error as GRE
from sklearn.model_selection import train_test_split
from skmatter.preprocessing import StandardFlexibleScaler
import pickle


lmax = 9
nmax = 6

atom_frames = read(
    "./benzenes.xyz", ":"
)  # all atom frames, containing benzene energies
frames = read("./ellipsoids.xyz", ":")  # ellipsoid frames

a1, a2, a3 = 4.0, 4.0, 0.5
for frame in frames:
    frame.arrays["c_diameter[1]"] = a1 * np.ones(len(frame))
    frame.arrays["c_diameter[2]"] = a2 * np.ones(len(frame))
    frame.arrays["c_diameter[3]"] = a3 * np.ones(len(frame))

AniSOAP_HYPERS = {
    "max_angular": lmax,
    "max_radial": nmax,
    "radial_basis_name": "gto",
    "subtract_center_contribution": True,
    "rotation_type": "quaternion",
    "rotation_key": "c_q",
    "cutoff_radius": 7.0,
    "radial_gaussian_width": 1.5,
    "basis_rcond": 1e-8,
    "basis_tol": 1e-3,
}


def main():
    """
    Things to check for:
    Where is the biggest slowdown? Is it the 5 nested for loops?
    How does this scale with number of frames inputted into power_spectrum?
    """
    import cProfile
    import pstats
    with cProfile.Profile() as pr:
        calculator = EllipsoidalDensityProjection(**AniSOAP_HYPERS)
        x_anisoap_raw = calculator.power_spectrum(frames)
    # Do work here for printing out stats (see mCoding video)
    # Either output in stdout or store a binary .prof, whichever is better.

if __name__ == "__main__":
    main()

