# %%
"""
Example 3: Machine-learning benzene with mixed All Atom and CG Representation.
============================================================
This example builds upon the previous example, and demonstrates how 
to create and evaluate mixed representations. 

1. How to read ellipsoidal frames from ``.xyz`` file.
2. How to convert ellipsoidal frames to AniSOAP vectors.
3. How to use these frames in machine learning models.
"""
# sphinx_gallery_thumbnail_number = 3

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
from sklearn.metrics import root_mean_squared_error as RMSE
from skmatter.preprocessing import StandardFlexibleScaler
import pickle

# %%
# Read the frames

# lmax = 9
# nmax = 6
lmax = 2
nmax = 2
rcut = 7.0

atom_frames = read(
    "./benzenes.xyz", ":"
)  # all atom frames, containing benzene energies
frames = read("./ellipsoids.xyz", ":")  # ellipsoid frames
energies = np.array([aframe.info["energy_pa"] for aframe in atom_frames])
energies = np.reshape(
    energies, (-1, 1)
)  # Turn energies into column vector, required for sklearn
plt.hist(energies, bins=100)
plt.xlabel("Loaded Energies, eV")
plt.show()


# %%
# Computing the AniSOAP Vectors
#
# * The ideal semiaxes for the ellipsoid are (4, 4, 0.5)

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
    "cutoff_radius": rcut,
    "radial_gaussian_width": 1.5,
    "basis_rcond": 1e-8,
    "basis_tol": 1e-3,
}

calculator = EllipsoidalDensityProjection(**AniSOAP_HYPERS)

x_anisoap_raw = calculator.power_spectrum(frames)

#%%
# Create an All Atom Representation with the same Hypers
SOAP_HYPERS = {
    "cutoff": {
        "radius": rcut,
        "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    },
    "density": {
        "type": "Gaussian",
        "width": 0.3,
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": lmax,
        "radial": {"type": "Gto", "max_radial": nmax},
    },
}

calculator = SoapPowerSpectrum(**SOAP_HYPERS)
descriptor_example = calculator.compute(atom_frames)
print("before: ", len(descriptor_example.keys))

descriptor_example = descriptor_example.keys_to_samples("center_type")
descriptor_example = descriptor_example.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
print("after: ", len(descriptor_example.keys))
x_soap_raw = metatensor.mean_over_samples(descriptor_example, sample_names=['atom', 'center_type'])
x_soap_raw = x_soap_raw.block().values.squeeze()

#%%
# Create a spherical CG SOAP Representation with the same Hypers
soapcg_desc = calculator.compute(frames)
soapcg_desc = soapcg_desc.keys_to_samples("center_type")
soapcg_desc = soapcg_desc.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
x_soapcg_raw = metatensor.mean_over_samples(soapcg_desc, sample_names=['atom', 'center_type'])
x_soapcg_raw = x_soapcg_raw.block().values.squeeze()
# %%
# Here, we do standard preparation of the data for machine learning.
#
# * Perform a train test split and standardization.
# * Note: Warnings below are from StandardFlexibleScaler.

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skmatter.decomposition import PCovR
pca = PCA(n_components=21)
pcovr = PCovR(n_components=21)
i_train, i_test = train_test_split(np.arange(len(frames)), train_size=0.9, shuffle=True)

y_train_scaler = StandardFlexibleScaler(column_wise=True).fit(energies[i_train])
y_train = y_train_scaler.transform(energies[i_train])
y_test_scaler = StandardFlexibleScaler(column_wise=True).fit(energies[i_test])
y_test = y_test_scaler.transform(energies[i_test])

x_train_anisoap_scaler = StandardFlexibleScaler(column_wise=False).fit(x_anisoap_raw[i_train])
x_train_anisoap = x_train_anisoap_scaler.transform(x_anisoap_raw[i_train])
x_train_anisoap = pcovr.fit_transform(x_train_anisoap, y_train)
x_test_anisoap_scaler = StandardFlexibleScaler(column_wise=False).fit(x_anisoap_raw[i_test])
x_test_anisoap = x_test_anisoap_scaler.transform(x_anisoap_raw[i_test])
x_test_anisoap = pcovr.fit_transform(x_test_anisoap, y_test)

x_train_soap_scaler = StandardFlexibleScaler(column_wise=False).fit(x_soap_raw[i_train])
x_train_soap = x_train_soap_scaler.transform(x_soap_raw[i_train])
x_train_soap = pcovr.fit_transform(x_train_soap, y_train)
x_test_soap_scaler = StandardFlexibleScaler(column_wise=False).fit(x_soap_raw[i_test])
x_test_soap = x_test_soap_scaler.transform(x_soap_raw[i_test])
x_test_soap = pcovr.fit_transform(x_test_soap, y_test)

x_train_soapcg_scaler = StandardFlexibleScaler(column_wise=False).fit(x_soapcg_raw[i_train])
x_train_soapcg = x_train_soapcg_scaler.transform(x_soapcg_raw[i_train])
x_train_soapcg = pcovr.fit_transform(x_train_soapcg, y_train)
x_test_soapcg_scaler = StandardFlexibleScaler(column_wise=False).fit(x_soapcg_raw[i_test])
x_test_soapcg = x_test_soapcg_scaler.transform(x_soapcg_raw[i_test])
x_test_soapcg = pcovr.fit_transform(x_test_soapcg, y_test)

# %%
# Input into a regularized linear regression machine learning model

from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge

# lr = RidgeCV(cv=5, alphas=np.logspace(-8, 2, 20), fit_intercept=True)
lr = KernelRidge(alpha=1e-8)
lr.fit(x_train_anisoap, y_train)
# print(f"{lr.alpha_=:.3f}")

# %%
# Model performance and Parity Plot
y_train_pred_anisoap = lr.predict(x_train_anisoap)
y_test_pred_anisoap = lr.predict(x_test_anisoap)
energies_train_pred_anisoap = y_train_scaler.inverse_transform(y_train_pred_anisoap.reshape(-1, 1))
energies_test_pred_anisoap = y_test_scaler.inverse_transform(y_test_pred_anisoap.reshape(-1, 1))

plt.figure(figsize=(8, 8))
plt.scatter(energies[i_train], energies_train_pred_anisoap, alpha=0.5)
plt.scatter(energies[i_test], energies_test_pred_anisoap, alpha=0.5)
plt.plot(
    [np.min(energies), np.max(energies)], [np.min(energies), np.max(energies)], "r--"
)
plt.xlabel("Per-atom Energies (eV)")
plt.ylabel("AniSOAP Predicted Per-atom Energies (eV)")
plt.legend(["Train", "Test", "y=x"])

print(f"AniSOAP Train R^2: {lr.score(x_train_anisoap, y_train):.3f}")
print(f"AniSOAP Train RMSE: {RMSE(energies[i_train], energies_train_pred_anisoap)}") 
print(f"AniSOAP Test R^2: {lr.score(x_test_anisoap, y_test):.3f}")
print(f"AniSOAP Test RMSE: {RMSE(energies[i_test], energies_test_pred_anisoap)}") 


# %%
lr.fit(x_train_soap, y_train)
y_train_pred_soap = lr.predict(x_train_soap)
y_test_pred_soap = lr.predict(x_test_soap)
energies_train_pred_soap = y_train_scaler.inverse_transform(y_train_pred_soap.reshape(-1, 1))
energies_test_pred_soap = y_test_scaler.inverse_transform(y_test_pred_soap.reshape(-1, 1))

plt.figure(figsize=(8, 8))
plt.scatter(energies[i_train], energies_train_pred_soap, alpha=0.5)
plt.scatter(energies[i_test], energies_test_pred_soap, alpha=0.5)
plt.plot(
    [np.min(energies), np.max(energies)], [np.min(energies), np.max(energies)], "r--"
)
plt.xlabel("Per-atom Energies (eV)")
plt.ylabel("SOAP Predicted Per-atom Energies (eV)")
plt.legend(["Train", "Test", "y=x"])

print(f"SOAP Train R^2: {lr.score(x_train_soap, y_train):.3f}")
print(f"SOAP Train RMSE: {RMSE(energies[i_train], energies_train_pred_soap)}") 
print(f"SOAP Test R^2: {lr.score(x_test_soap, y_test):.3f}")
print(f"SOAP Test RMSE: {RMSE(energies[i_test], energies_test_pred_soap)}") 
# %%
lr.fit(x_train_soapcg, y_train)
y_train_pred_soapcg = lr.predict(x_train_soapcg)
y_test_pred_soapcg = lr.predict(x_test_soapcg)
energies_train_pred_soapcg = y_train_scaler.inverse_transform(y_train_pred_soapcg.reshape(-1, 1))
energies_test_pred_soapcg = y_test_scaler.inverse_transform(y_test_pred_soapcg.reshape(-1, 1))

plt.figure(figsize=(8, 8))
plt.scatter(energies[i_train], energies_train_pred_soapcg, alpha=0.5)
plt.scatter(energies[i_test], energies_test_pred_soapcg, alpha=0.5)
plt.plot(
    [np.min(energies), np.max(energies)], [np.min(energies), np.max(energies)], "r--"
)
plt.xlabel("Per-atom Energies (eV)")
plt.ylabel("SOAP CG Predicted Per-atom Energies (eV)")
plt.legend(["Train", "Test", "y=x"])

print(f"SOAP_CG Train R^2: {lr.score(x_train_soapcg, y_train):.3f}")
print(f"SOAP_CG Train RMSE: {RMSE(energies[i_train], energies_train_pred_soapcg)}") 
print(f"SOAP_CG Test R^2: {lr.score(x_test_soapcg, y_test):.3f}")
print(f"SOAP_CG Test RMSE: {RMSE(energies[i_test], energies_test_pred_soapcg)}") 

#%%
def mixed_kernel(X1, X2):
    # Each are rows of a concatenated matrix
    x_soap1, x_anisoap1 = X1[:21], X1[21:]
    x_soap2, x_anisoap2 = X2[:21], X2[21:]
    return np.dot(x_soap1, x_soap2) + np.dot(x_anisoap1, x_anisoap2)
# %%
# Create mixed AniSOAP/SOAP representation
from skmatter.preprocessing import KernelNormalizer
alpha = 1
Kaa = x_train_soap @ x_train_soap.T
Kcg = x_train_anisoap @ x_train_anisoap.T
Kicg = x_train_soapcg @ x_train_soapcg.T
kernel_normalizer_aa = KernelNormalizer()
kernel_normalizer_cg = KernelNormalizer()
kernel_normalizer_aa.fit(Kaa)
kernel_normalizer_cg.fit(Kcg)
Kaa_norm = kernel_normalizer_aa.transform(Kaa)
Kcg_norm = kernel_normalizer_cg.transform(Kcg)
Kmix = alpha * Kaa_norm + (1 - alpha) * Kcg_norm

Kaa_test = x_test_soap @ x_train_soap.T
Kcg_test = x_test_anisoap @ x_train_anisoap.T
Kaa_test = kernel_normalizer_aa.transform(Kaa_test)
Kcg_test = kernel_normalizer_cg.transform(Kcg_test)
Kmix_test = alpha * Kaa_test + (1 - alpha) * Kcg_test

lr = KernelRidge(alpha=1e-8, kernel="precomputed")

lr.fit(Kmix, y_train)

y_train_pred_mix = lr.predict(Kmix)
y_test_pred_mix = lr.predict(Kmix_test)
energies_train_pred_mix = y_train_scaler.inverse_transform(y_train_pred_mix.reshape(-1, 1))
energies_test_pred_mix = y_test_scaler.inverse_transform(y_test_pred_mix.reshape(-1, 1))
print(f"{alpha=}")
print(f"MixedRep Train R^2: {lr.score(Kmix, y_train):.3f}")
print(f"MixedRep Train RMSE: {RMSE(energies[i_train], energies_train_pred_mix)}") 
print(f"MixedRep Test R^2: {lr.score(Kmix_test, y_test):.3f}")
print(f"MixedRep Test RMSE: {RMSE(energies[i_test], energies_test_pred_mix)}") 
# %%
from tqdm.auto import tqdm
from sklearn.decomposition import PCA 
train_rmses = []
test_rmses = []
alphas = np.linspace(0, 1, 101)
for alpha in tqdm(alphas):
    x_train_mix_raw = np.hstack((alpha * x_train_soap, (1-alpha) * x_train_anisoap))
    x_train_mix = StandardFlexibleScaler(column_wise=False).fit_transform(x_train_mix_raw)

    x_test_mix_raw = np.hstack((alpha * x_test_soap, (1-alpha) * x_test_anisoap))
    # x_test_mix_raw = x_test_mix_raw[:, x_test_mix_raw.var(axis=0)>1e-12]
    x_test_mix = StandardFlexibleScaler(column_wise=False).fit_transform(x_test_mix_raw)

    lr.fit(x_train_mix, y_train)
    y_train_pred_mix = lr.predict(x_train_mix)
    y_test_pred_mix = lr.predict(x_test_mix)
    energies_train_pred_mix = y_train_scaler.inverse_transform(y_train_pred_mix.reshape(-1, 1))
    energies_test_pred_mix = y_test_scaler.inverse_transform(y_test_pred_mix.reshape(-1, 1))
    train_rmses.append(RMSE(energies[i_train], energies_train_pred_mix))
    test_rmses.append(RMSE(energies[i_test], energies_test_pred_mix))
# %%
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = "Helvetica"
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.size'] = 14

mpl.rcParams['mathtext.fontset'] = "custom"
mpl.rcParams['svg.fonttype'] = 'none'
plt.plot(alphas, np.array(train_rmses), linewidth=2.0)
plt.xlabel("$\lambda$ (0 is CG AniSOAP, 1 is All-Atom SOAP)")
plt.ylabel("RMSE of Energy Prediction (eV/atom)")
plt.savefig("bananaplot_CGAniSOAP_CGSOAP.svg")

# %%
from tqdm.auto import tqdm
from sklearn.decomposition import PCA 
train_rmses2 = []
test_rmses2 = []
alphas = np.linspace(0, 1, 101)
for alpha in tqdm(alphas):
    x_train_mix2_raw = np.hstack((alpha * x_train_soap, (1-alpha) * x_train_soapcg))
    x_train_mix2 = StandardFlexibleScaler(column_wise=False).fit_transform(x_train_mix2_raw)

    x_test_mix_raw2 = np.hstack((alpha * x_test_soap, (1-alpha) * x_test_soapcg))
    # x_test_mix_raw = x_test_mix_raw[:, x_test_mix_raw.var(axis=0)>1e-12]
    x_test_mix2 = StandardFlexibleScaler(column_wise=False).fit_transform(x_test_mix_raw2)

    lr.fit(x_train_mix2, y_train)
    y_train_pred_mix2 = lr.predict(x_train_mix2)
    y_test_pred_mix2 = lr.predict(x_test_mix2)
    energies_train_pred_mix2 = y_train_scaler.inverse_transform(y_train_pred_mix2.reshape(-1, 1))
    energies_test_pred_mix2 = y_test_scaler.inverse_transform(y_test_pred_mix2.reshape(-1, 1))
    # print(f"{alpha=}")
    # print(f"MixedRep Train R^2: {lr.score(x_train_mix2, y_train):.3f}")
    # print(f"MixedRep Train RMSE: {RMSE(energies[i_train], energies_train_pred_mix)}") 
    # print(f"MixedRep Test R^2: {lr.score(x_test_mix, y_test):.3f}")
    # print(f"MixedRep Test RMSE: {RMSE(energies[i_test], energies_test_pred_mix)}") 
    train_rmses2.append(RMSE(energies[i_train], energies_train_pred_mix2))
    test_rmses2.append(RMSE(energies[i_test], energies_test_pred_mix2))
# %%
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = "Helvetica"
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.size'] = 14

mpl.rcParams['mathtext.fontset'] = "custom"
mpl.rcParams['svg.fonttype'] = 'none'
plt.plot(alphas, np.array(train_rmses2), linewidth=2.0)
plt.xlabel("$\lambda$ (0 is CG AniSOAP, 1 is All-Atom SOAP)")
plt.ylabel("RMSE of Energy Prediction (eV/atom)")
plt.savefig("bananaplot_CGAniSOAP_CGSOAP.svg")
# %%

