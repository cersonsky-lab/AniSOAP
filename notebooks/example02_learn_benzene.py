#%%
"""
Example 2: Machine-learning benzene energies.
============================================================
This example demonstrates:

1. How to read ellipsoidal frames from ``.xyz`` file.
2. How to convert ellipsoidal frames to AniSOAP vectors.
3. How to use these frames in machine learning models.
"""

import metatensor
import numpy as np
from anisoap.representations import EllipsoidalDensityProjection
from anisoap.utils import ClebschGordanReal, cg_combine, standardize_keys
from ase.io import read
from matplotlib import pyplot as plt
from matplotlib import rc
from rascaline import SoapPowerSpectrum
from sklearn.decomposition import PCA
from skmatter.metrics import global_reconstruction_error as GRE
from sklearn.model_selection import train_test_split
from skmatter.preprocessing import StandardFlexibleScaler
import pickle

# %%
# Read the frames

lmax = 9
nmax = 6

atom_frames = read("benzenes.xyz", ":")    # all atom frames, containing benzene energies
frames = read("ellipsoids.xyz", ":")    # ellipsoid frames
energies = np.array([aframe.info["energy_pa"] for aframe in atom_frames])
energies = np.reshape(energies, (-1, 1))   # Turn energies into column vector, required for sklearn
plt.hist(energies, bins=100)
plt.xlabel("Loaded Energies, eV")
plt.show()


# %% 
# Computing the AniSOAP Vectors
# The ideal semiaxes for the ellipsoid are (4, 4, 0.5)

a1, a2, a3 = 4., 4., 0.5
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

calculator = EllipsoidalDensityProjection(**AniSOAP_HYPERS)

x_anisoap_raw = calculator.power_spectrum(frames, show_progress=True, rust_moments=True)

# %%
# Here, we do standard preparation of the data for machine learning.
# Perform a train test split and standardization.

from sklearn.model_selection import train_test_split

i_train, i_test = train_test_split(np.arange(len(frames)), train_size=0.9, shuffle=True)
x_train_scaler = StandardFlexibleScaler(column_wise=False).fit(x_anisoap_raw[i_train])
x_train = x_train_scaler.transform(x_anisoap_raw[i_train])
y_train_scaler = StandardFlexibleScaler(column_wise=True).fit(energies[i_train])
y_train = y_train_scaler.transform(energies[i_train])

x_test_scaler = StandardFlexibleScaler(column_wise=False).fit(x_anisoap_raw[i_test])
x_test = x_test_scaler.transform(x_anisoap_raw[i_test])
y_test_scaler = StandardFlexibleScaler(column_wise=True).fit(energies[i_test])
y_test = y_test_scaler.transform(energies[i_test])

# %% 
# Input into a regularized linear regression machine learning model

from sklearn.linear_model import RidgeCV

lr = RidgeCV(cv=5, alphas=np.logspace(-8, 2, 20), fit_intercept=True)
lr.fit(x_train, y_train)

#%%
print(i_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(lr.alpha_)
# %%
# Model performance and Parity Plot
print(f"Train R^2: {lr.score(x_train, y_train):.3f}")
print(f"Test R^2: {lr.score(x_test, y_test):.3f}")


# x_train, x_test = x_anisoap[i_train], x_anisoap[i_test]
# y_train, y_test = y[i_train], y[i_test]

# # Standardize the AniSOAP vector and the energies
# x_anisoap_scalar = StandardFlexibleScaler(column_wise=False).fit(x_anisoap_raw)
# x_anisoap = x_anisoap_scalar.transform(x_anisoap_raw)

# # First, reshape energies to be a column vector, then standardize.
# y_scalar = StandardFlexibleScaler(column_wise=True).fit(energies)
# y = y_scalar.transform(energies)

#%%
# We can use the standardized ``x_anisoap`` to as input into a regularized 
# linear regression model
# from sklearn.linear_model import RidgeCV
# from sklearn.model_selection import train_test_split

# i_train, i_test = train_test_split(np.arange(len(x_anisoap)), train_size=0.9)

# x_train, x_test = x_anisoap[i_train], x_anisoap[i_test]
# y_train, y_test = y[i_train], y[i_test]

# lr = RidgeCV(cv=5, alphas=np.logspace(-8, 2, 20), fit_intercept=False)
# lr.fit(x_train, y_train)

#%%
# # Model performance and Parity Plot
# print(f"Train R^2: {lr.score(x_train, y_train):.3f}")
# print(f"Test R^2: {lr.score(x_test, y_test):.3f}")


# %% 
# Now, we compute the all-atom SOAP vectors 