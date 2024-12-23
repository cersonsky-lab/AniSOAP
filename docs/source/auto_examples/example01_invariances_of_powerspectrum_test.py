#%%
"""
Example 1: Creating AniSOAP vectors from ellipsoidal frames.
============================================================
This example demonstrates:

1. How to read ellipsoidal frames from ``.xyz`` file
2. How to convert ellipsoidal frames to AniSOAP vectors
3. How to create ellipsoidal frames with ``ase.Atoms``
"""

import sys
import warnings
import metatensor
from itertools import product
from ase.io import read
from ase import Atoms

import numpy as np
from metatensor import (
    Labels,
    TensorBlock,
    TensorMap,
)

from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm
from skmatter.preprocessing import StandardFlexibleScaler

from anisoap.representations.radial_basis import (
    GTORadialBasis,
    MonomialBasis,
)
from anisoap.representations.ellipsoidal_density_projection import (
    EllipsoidalDensityProjection,
)

import matplotlib.pyplot as plt



# %%
# Read the first two frames of ellipsoids.xyz, which represent coarse-grained benzene molecules.

frames = read("ellipsoids.xyz", "0:2")
frames_translation = read("ellipsoids.xyz", "0:2")
frames_rotation = read("ellipsoids.xyz", "0:2")

print(f"{len(frames)=}")   # a list of atoms objects
print(f"{frames[0].arrays=}")

#%% 
# In this case, the xyz file did not store ellipsoid dimension information. 
# 
# We will add this information here.

for frame in frames:
    frame.arrays["c_diameter[1]"] = np.ones(len(frame)) * 3.
    frame.arrays["c_diameter[2]"] = np.ones(len(frame)) * 3.
    frame.arrays["c_diameter[3]"] = np.ones(len(frame)) * 1.

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
# Create the AniSOAP vector (i.e. the power spectrum).
power_spectrum = calculator.power_spectrum(frames)
plt.plot(power_spectrum.T)
plt.legend(["frame[0] power spectrum", "frame[1] power spectrum"])
plt.show()

# %% 
# Here we will demonstrate translation invariance.
# 
# Translation vector is used to demonstrate the power spectrum of ellipsoidal representations are invariant of translation in positions.
print("Old Positions:", frames[0].get_positions(), frames[1].get_positions())
translation_vector = np.array([2.0, 2.0, 2.0])
for frame in frames:
    frame.set_positions(frame.get_positions() + translation_vector)
print("New Positions:", frames[0].get_positions(), frames[1].get_positions())
power_spectrum_translated = calculator.power_spectrum(frames)
print(f"{np.allclose(power_spectrum, power_spectrum_translated)=}")

# %% 
# Here, we demonstrate rotational invariance, rotating all ellipsoids by the same amount.
print("Old Orientations:", frames[0].arrays["c_q"], frames[1].arrays["c_q"])

quaternion = [1, 2, 0, -3]   # random rotation
q_rotation = R.from_quat(quaternion, scalar_first=True)   
for frame in frames:
    frame.arrays["c_q"] = R.as_quat(
        q_rotation * R.from_quat(frame.arrays["c_q"], scalar_first=True),
        scalar_first=True,
    )
print("New Orientations:", frames[0].arrays["c_q"], frames[1].arrays["c_q"])

power_spectrum_rotation = calculator.power_spectrum(frames)
print(f"{np.allclose(power_spectrum, power_spectrum_rotation, rtol=1e-2, atol=1e-2)=}")

# %%
# Here's how to create ellipsoidal frames. In this example:
# 
# * Each frame contains 2-3 ellipsoids, with periodic boundary conditions.
# * The quaternions(``c_q``) and particle dimensions(``c_diameter[i]``) cannot be passed into the Atoms constructor.
# * They are attached as data in the Atoms.arrays dictionary.
# * I just made up arbitrary postions and orientations. Quaternions should be in (w,x,y,z) format.
# * In reality you would choose positions and orientations based on some underlying atomistic model.
frame1 = Atoms(symbols='XX', 
               positions=np.array([[0., 0., 0.], [2.5, 3., 2.]]),
               cell=np.array([5., 5., 5.,]), 
               pbc=True)
frame1.arrays["c_q"] = np.array([[0., 1., 0., 0.], [0., 0., 1., 0]])
frame1.arrays["c_diameter[1]"] = np.array([3., 3.])
frame1.arrays["c_diameter[2]"] = np.array([3., 3.])
frame1.arrays["c_diameter[3]"] = np.array([1., 1.])

frame2 = Atoms(symbols='XXX', 
               positions = np.array([[0., 1., 2.], [2., 3., 4.], [5., 5., 1.]]),
               cell=[10., 10., 10.,], 
               pbc=True)
frame2.arrays["c_q"] = np.array([[0., 1., 0., 0.], [0., 0., 1., 0], [0., 0., 0.707, 0.707]])
frame2.arrays["c_diameter[1]"] = np.array([3., 3., 3.])
frame2.arrays["c_diameter[2]"] = np.array([3., 3., 3.])
frame2.arrays["c_diameter[3]"] = np.array([1., 1., 1.])

frames = [frame1, frame2]

#%%
# You can then use ``ase.io.write()``/``ase.io.read()`` to save/load these frames for later use.