import numpy as np
from ase import Atoms
import pytest
from numpy.testing import assert_allclose
from anisoap.representations import EllipsoidalDensityProjection
import metatensor

from scipy.special import sph_harm


class TestGaussianConvergence:
    sigmas = np.array([1.0, 2.0, 3.0])

    @pytest.mark.parametrize("sigma", sigmas)
    def test_single_atom_isotropic_convergence(self, sigma):
        """
        Test that coefficients are correct, such that the approximation using these coefficients in the expansion
        can reasonably recreate the original atomic gaussian (in this case, an unnormalized gaussian with sigma=1)
        """

        frame = Atoms(positions=np.array([[0, 0, 0]]), numbers=[0])
        frame.arrays["quaternions"] = np.array([[1, 0, 0, 0]])

        frame.arrays[r"c_diameter[1]"] = 2 * sigma * np.ones(1)
        frame.arrays[r"c_diameter[2]"] = 2 * sigma * np.ones(1)
        frame.arrays[r"c_diameter[3]"] = 2 * sigma * np.ones(1)
        frames = [frame]

        # An rgw that's a little bigger than the atom gaussian sigma converges well and is sufficient for this test.
        rgw = sigma + 0.5

        max_radials = range(10)
        errs = []
        for max_radial in max_radials:
            HYPER_PARAMETERS = {
                "max_angular": 1,
                "max_radial": max_radial,
                "radial_basis_name": "gto",
                "rotation_type": "quaternion",
                "rotation_key": "quaternions",
                "cutoff_radius": 1.0,
                "radial_gaussian_width": rgw,
                "basis_rcond": 1e-14,
                "basis_tol": 1e-1,
            }
            representation = EllipsoidalDensityProjection(**HYPER_PARAMETERS)
            descriptor_raw = representation.transform(frames, normalize=True)
            descriptor = metatensor.operations.sum_over_samples(
                descriptor_raw, sample_names="center"
            )

            def evaluate_bases(r):
                def real_sphharm(m, l, theta, phi):
                    if m < 0:
                        return (
                            np.sqrt(2)
                            * (-1.0) ** m
                            * np.imag(sph_harm(np.abs(m), l, theta, phi))
                        )
                    elif m == 0:
                        # this is real anyway, just doing this so we don't get an annoying "discarding imaginary warning"
                        return np.real(sph_harm(m, l, theta, phi))
                    else:
                        return (
                            np.sqrt(2)
                            * (-1.0) ** m
                            * np.real(sph_harm(m, l, theta, phi))
                        )

                bases = descriptor.copy()
                for key, block in bases.items():
                    l = key["angular_channel"]
                    for m in block.components[0][
                        "spherical_component_m"
                    ]:  # same as range(-l, l+1)
                        # evaluated at rhat = (theta, phi) = (0, 0) -- ok if gaussian is isotropic.
                        ylm = real_sphharm(m, l, 0, 0)
                        for n in block.properties["n"]:
                            rnl = representation.radial_basis.get_basis(r).flatten()[n]
                            block.values[0, m, n] = ylm * rnl
                return bases

        # Now, we perform our reconstruction
        approx = []
        r_mesh = np.linspace(0, 5, 100)
        length_norm = (sigma**3 * (2.0 * np.pi) ** (3.0 / 2.0)) ** -1.0
        actual = length_norm * np.exp(-(r_mesh**2) / (2 * sigma**2))

        for r in r_mesh:
            # For now, have to do an inefficient pointwise evaluation, unfortunately.
            bases = evaluate_bases(np.array([r]))
            approx.append(np.sum(bases.block(0).values * descriptor.block(0).values))
        approx = np.array(approx)
        errs.append(np.sum((approx - actual) ** 2) / len(approx))

        # assert that errors are monotonically decreasing. As we increase the number of terms in the expansion,
        # we should get a better and better approximation.
        assert np.all(errs[:-1] >= errs[1:])

        # as long as each element differs by a (somewhat) small absolute amount, we're good.
        assert_allclose(approx, actual, atol=1e-03)
