# %%
"""
Example 1: Creating AniSOAP vectors from ellipsoidal frames.
============================================================
This example demonstrates:

1. How to read ellipsoidal frames from ``.xyz`` file.
2. How to convert ellipsoidal frames to AniSOAP vectors, via the power_spectrum convenience method and via manual calculations and combinations of expansion coefficients. We also demonstrate the Translational and Rotational invariance of these representations.
3. How to create ellipsoidal frames with ``ase.Atoms``.
"""
from ase.io import read
from ase import Atoms

import numpy as np

from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm
from skmatter.preprocessing import StandardFlexibleScaler

from anisoap.representations.ellipsoidal_density_projection import (
    EllipsoidalDensityProjection,
)

import matplotlib.pyplot as plt


# %%
# Read the first two frames of ellipsoids.xyz, which represent coarse-grained benzene molecules.

frames = read("./ellipsoids.xyz", "0:2")
frames_translation = read("./ellipsoids.xyz", "0:2")
frames_rotation = read("./ellipsoids.xyz", "0:2")

print(f"{len(frames)=}")  # a list of atoms objects
print(f"{frames[0].arrays=}")

# %%
# In this case, the xyz file did not store ellipsoid dimension information.
#
# We will add this information here.

for frame in frames:
    frame.arrays["c_diameter[1]"] = np.ones(len(frame)) * 3.0
    frame.arrays["c_diameter[2]"] = np.ones(len(frame)) * 3.0
    frame.arrays["c_diameter[3]"] = np.ones(len(frame)) * 1.0

print(f"{frames[0].arrays=}")
print(f"{frames[1].arrays=}")

# %%
# Specify the hypers to create AniSOAP vector.

lmax = 5
nmax = 3

AniSOAP_HYPERS = {
    "max_angular": lmax,
    "max_radial": nmax,
    "radial_basis_name": "gto",
    "rotation_type": "quaternion",
    "rotation_key": "c_q",
    "cutoff_radius": 7.0,
    "radial_gaussian_width": 1.5,
    "basis_rcond": 1e-8,
    "basis_tol": 1e-4,
}
calculator = EllipsoidalDensityProjection(**AniSOAP_HYPERS)

# %%
# First, we demonstrate how to create the AniSOAP vector (i.e. the power spectrum) using a convenience method.
power_spectrum = calculator.power_spectrum(frames)
plt.plot(power_spectrum.T)
plt.legend(["frame[0] power spectrum", "frame[1] power spectrum"])
plt.show()

# %%
# Under the hood, the `power_spectrum` convenience method first calculates the expansion coefficients,
# then performs a Clebsch-Gordan product on these coefficients to obtain the power spectrum.
# This requires importing some utilities first.
from anisoap.utils.metatensor_utils import (
    ClebschGordanReal,
    cg_combine,
    standardize_keys,
)
import metatensor

mvg_coeffs = calculator.transform(frames)
mvg_nu1 = standardize_keys(mvg_coeffs)  # standardize the metadata naming schemes
# Create an object that stores Clebsch-Gordan coefficients for a certain lmax:
mycg = ClebschGordanReal(lmax)

# Combines the mvg_nu1 with itself using the Clebsch-Gordan coefficients.
# This combines the angular and radial components of the sample.
mvg_nu2 = cg_combine(
    mvg_nu1,
    mvg_nu1,
    clebsch_gordan=mycg,
    lcut=0,
    other_keys_match=["types_center"],
)

# mvg_nu2 is the unaggregated form of the AniSOAP descriptor and is what is returned by `power_spectrum` when mean_over_samples=False.
# Typically, we want an aggregated representation on a per-frame basis, rather than an unaggregated per-frame-per-particle representation.
mvg_nu2_avg = metatensor.mean_over_samples(mvg_nu2, sample_names="center")
x_asoap_raw = mvg_nu2_avg.block().values.squeeze()
# This (n_samples x n_features) feature matrix can then be fed into an ML learning architecture.
# This is equivalent to the output of the convenience method used above.
if np.allclose(x_asoap_raw, power_spectrum):
    print("The two representations are equivalent")
# %%
# Here we will demonstrate translation invariance.
#
# A translation vector is used to demonstrate that the power spectrum of ellipsoidal representations is invariant to translation in positions.
print("Old Positions:", frames[0].get_positions(), frames[1].get_positions())
translation_vector = np.array([2.0, 2.0, 2.0])
for frame in frames:
    frame.set_positions(frame.get_positions() + translation_vector)
print("New Positions:", frames[0].get_positions(), frames[1].get_positions())
power_spectrum_translated = calculator.power_spectrum(frames)

if np.allclose(power_spectrum, power_spectrum_translated):
    print("Power spectrum has translational invariance!")
else:
    print("Power spectrum has no translational invariance")

# %%
# Here, we demonstrate rotational invariance, rotating all ellipsoids by the same amount.
print("Old Orientations:", frames[0].arrays["c_q"], frames[1].arrays["c_q"])

quaternion = [1, 2, 0, -3]  # random rotation
q_rotation = R.from_quat(quaternion, scalar_first=True)
for frame in frames:
    frame.arrays["c_q"] = R.as_quat(
        q_rotation * R.from_quat(frame.arrays["c_q"], scalar_first=True),
        scalar_first=True,
    )
print("New Orientations:", frames[0].arrays["c_q"], frames[1].arrays["c_q"])

power_spectrum_rotation = calculator.power_spectrum(frames)
if np.allclose(power_spectrum, power_spectrum_rotation, rtol=1e-2, atol=1e-2):
    print("Power spectrum has rotation invariance (with lenient tolerances)")
else:
    print("Power spectrum has no rotation invariance")

# %%
# Here's how to create ellipsoidal frames. In this example:
#
# * Each frame contains 2-3 ellipsoids, with periodic boundary conditions.
# * The quaternions(``c_q``) and particle dimensions(``c_diameter[i]``) cannot be passed into the Atoms constructor.
# * They are attached as data in the Atoms.arrays dictionary.
# * I just made up arbitrary postions and orientations. Quaternions should be in (w,x,y,z) format.
# * In reality you would choose positions and orientations based on some underlying atomistic model.
frame1 = Atoms(
    symbols="XX",
    positions=np.array([[0.0, 0.0, 0.0], [2.5, 3.0, 2.0]]),
    cell=np.array(
        [
            5.0,
            5.0,
            5.0,
        ]
    ),
    pbc=True,
)
frame1.arrays["c_q"] = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0]])
frame1.arrays["c_diameter[1]"] = np.array([3.0, 3.0])
frame1.arrays["c_diameter[2]"] = np.array([3.0, 3.0])
frame1.arrays["c_diameter[3]"] = np.array([1.0, 1.0])

frame2 = Atoms(
    symbols="XXX",
    positions=np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [5.0, 5.0, 1.0]]),
    cell=[
        10.0,
        10.0,
        10.0,
    ],
    pbc=True,
)
frame2.arrays["c_q"] = np.array(
    [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0], [0.0, 0.0, 0.707, 0.707]]
)
frame2.arrays["c_diameter[1]"] = np.array([3.0, 3.0, 3.0])
frame2.arrays["c_diameter[2]"] = np.array([3.0, 3.0, 3.0])
frame2.arrays["c_diameter[3]"] = np.array([1.0, 1.0, 1.0])

frames = [frame1, frame2]

# %%
# You can then use ``ase.io.write()``/``ase.io.read()`` to save/load these frames for later use.
