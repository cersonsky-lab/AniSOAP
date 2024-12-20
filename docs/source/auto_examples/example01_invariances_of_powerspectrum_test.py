"""
Example 1: Creating AniSOAP vectors from ellipsoidal frames.
============================================================
This example demonstrates:

1. How to create ellipsoidal frames

2. How to read ellipsoidal frame from xyz

3. How to convert ellipsoidal frames to AniSOAP vectors
"""

import sys
import warnings
import metatensor
from itertools import product
from ase.io import read

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
# This is a section header -- Read Ellipsoidal Frames 
# ---------------------------------------------------

ell_frames = read("ellipsoids.xyz", "0:2")
ell_frames_translation = read("ellipsoids.xyz", "0:2")
ell_frames_rotation = read("ellipsoids.xyz", "0:2")

# This is to make sure the ell_frames list calls c_diameter[] rather than c_diameter and to update the diameters of ellipsoids to be 3,3, and 1.


def update_diameters_and_variablename(frames):
    for frame in frames:
        for i in range(1, 4):
            old = f"c_diameter{i}"
            new = f"c_diameter[{i}]"
            if old in frame.arrays:
                frame.arrays[new] = frame.arrays[old]
            frame.arrays[new] = np.ones(len(frame)) * (3.0 if i < 3 else 1.0)


update_diameters_and_variablename(ell_frames)
update_diameters_and_variablename(ell_frames_translation)
update_diameters_and_variablename(ell_frames_rotation)
print("Hello, world!")
plt.plot(np.sin(np.linspace(0, 2*np.pi)))
