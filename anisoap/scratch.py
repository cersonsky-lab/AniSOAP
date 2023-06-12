from anisoap.representations import EllipsoidalDensityProjection
from anisoap.utils import standardize_keys, ClebschGordanReal, cg_combine
import equistore
from ase.io import read, write
import numpy as np
from skmatter.preprocessing import StandardFlexibleScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV

frames = read('/Users/alin62/Documents/Research/2023_Ellipsoidal_SOAP/ellipsoid_frames.xyz', ':')
atom_frames = read('/Users/alin62/Documents/Research/2023_Ellipsoidal_SOAP/atom_frames.xyz', ':')
energies = np.array([frame.info['energy_eV'] for frame in atom_frames]).reshape(-1,1)

a1 = 4.6
a3 = 1.4

l_max = 6
representation = EllipsoidalDensityProjection(max_angular=l_max,
                                              radial_basis_name='gto',
                                              rotation_type='quaternion',
                                              rotation_key='c_q',
                                              cutoff_radius=8,
                                              radial_gaussian_width=5.5)

for frame in frames:
    frame.arrays["c_diameter[1]"] = a1 * np.ones(len(frame))
    frame.arrays["c_diameter[2]"] = a1 * np.ones(len(frame))
    frame.arrays["c_diameter[3]"] = a3 * np.ones(len(frame))
    frame.arrays["quaternions"] = frame.arrays['c_q']

# Calculate the transform with the normalization!
rep_raw = representation.transform(frames, show_progress=False, normalize=True)
rep = equistore.operations.mean_over_samples(rep_raw, sample_names="center")


aniso_nu1 = standardize_keys(rep)
mycg = ClebschGordanReal(l_max)
aniso_nu2 = cg_combine(
    aniso_nu1,
    aniso_nu1,
    clebsch_gordan=mycg,
    lcut=0,
    other_keys_match=["species_center"],
)

x_raw = aniso_nu2.block().values.squeeze()
x_raw = x_raw[:, x_raw.var(axis=0)>1E-12]

x_scaler = StandardFlexibleScaler(column_wise=False).fit(x_raw)   # For normalized, we do a noncolumnwise shift-rescale
y_scaler = StandardFlexibleScaler(column_wise=True).fit(energies)

x = x_scaler.transform(x_raw)
y = y_scaler.transform(energies)

lr = RidgeCV(cv=5, alphas=np.logspace(-8, 2, 20), fit_intercept=False)
i_train, i_test = train_test_split(np.arange(x.shape[0]), shuffle=True, test_size=0.1, random_state=1)

lr.fit(x[i_train], y[i_train])
print("train score:", lr.score(x[i_train], y[i_train]), "\ntest score:", lr.score(x[i_test], y[i_test]))

plt.plot(x_raw.T)
plt.title("Normalized Representation")
plt.show()

# Calculate the transform withOUT the normalization!
rep_raw = representation.transform(frames, show_progress=False, normalize=False)
rep = equistore.operations.mean_over_samples(rep_raw, sample_names="center")


aniso_nu1 = standardize_keys(rep)
mycg = ClebschGordanReal(l_max)
aniso_nu2 = cg_combine(
    aniso_nu1,
    aniso_nu1,
    clebsch_gordan=mycg,
    lcut=0,
    other_keys_match=["species_center"],
)

x_raw = aniso_nu2.block().values.squeeze()
x_raw = x_raw[:, x_raw.var(axis=0)>1E-12]

# x_scaler = StandardFlexibleScaler(column_wise=True).fit(x_raw)
x_scaler = StandardFlexibleScaler(column_wise=True).fit(x_raw)     # For nonnormalized, we do a columnwise shift-rescale
y_scaler = StandardFlexibleScaler(column_wise=True).fit(energies)

x = x_scaler.transform(x_raw)
y = y_scaler.transform(energies)

lr = RidgeCV(cv=5, alphas=np.logspace(-8, 2, 20), fit_intercept=False)
i_train, i_test = train_test_split(np.arange(x.shape[0]), shuffle=True, test_size=0.1, random_state=1)

lr.fit(x[i_train], y[i_train])
print("train score:", lr.score(x[i_train], y[i_train]), "\ntest score:", lr.score(x[i_test], y[i_test]))
plt.plot(x_raw.T)
plt.title("UNnormalized Representation")
plt.show()




# IGNORE EVERYTHING BELOW
# aniso_nu1 = standardize_keys(rep_normalized)
# mycg = ClebschGordanReal(l_max)
# aniso_nu2 = cg_combine(
#     aniso_nu1,
#     aniso_nu1,
#     clebsch_gordan=mycg,
#     lcut=0,
#     other_keys_match=["species_center"],
# )
#
# x_raw_normalized = aniso_nu2.block().values.squeeze()
# x_raw_normalized = x_raw_normalized[:, x_raw_normalized.var(axis=0)>1e-12]
#
# plt.plot(x_raw.T)
# plt.show()
#
# plt.plot(x_raw_normalized.T)
# plt.show()

# x_scaler = StandardFlexibleScaler(column_wise=True).fit(
#     x_raw
# )
# x = x_scaler.transform(x_raw)
# x.shape


# def normalize_basis(self, features: TensorMap):
#     """
#     In each tensor block within the features:
#     Normalize each element of the tensor block's values.
#     Here, the normalization constant corresponding to alpha, n, l, m is given by:
#     .. math::
#         N(\alpha, l, m, n) = (\frac{8\alpha}{\pi})^{3/4} * [\frac{(8\alpha)^{l+m+n} * l! * m! * n!}{(2l)! * (2m)! * (2n)!}]^{1/2}]
#     Which was found in equation 2 of this paper: http://www.diva-portal.org/smash/get/diva2:282089/fulltext01
#
#     Here, the normalization constant is simply the max corresponding to the (n,l,m) gto basis function.
#
#     My current implementation is extremely inefficient as it recalculates
#     Args:
#         features: a tensormap containing all blocks
#
#     Returns:
#         normalized_features, a tensormap containing blocks whose radial bases are normalized
#     """
#
#     normalized_features = features.copy()
#     factorial = np.math.factorial
#     alpha = self.radial_basis.hypers['radial_gaussian_width']
#     for block in normalized_features.blocks():
#         # Need to just loop through the component dimension and property dimension:
#         l = block.components[0].asarray().flatten()[-1]
#         for i in range(block.values.shape[0]):
#             for j in range(block.values.shape[1]):
#                 for k in range(block.values.shape[2]):
#                     m = block.components[0].asarray().flatten()[j]
#                     n = block.properties[k][0]
#                     norm_const = (2 * alpha / np.pi) ** (3 / 4) * (
#                                 (8 * alpha) ** (l + m + n) * factorial(l) * factorial(m) * factorial(n) / (
#                                     factorial(2 * l) * factorial(2 * m) * factorial(2 * n))) ** (1 / 2)
#                     block.values[i, j, k] *= norm_const
#
#     return normalized_features