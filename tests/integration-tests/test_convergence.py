import metatensor.torch as mts
import torch
import pytest
import numpy as np

from ase import Atoms
from numpy.testing import assert_allclose
from scipy.special import sph_harm_y

from anisoap.representations import EllipsoidalDensityProjection


class TestGaussianConvergence:
    sigmas = [1.0, 2.0, 3.0]

    @pytest.mark.parametrize("sigma", sigmas)
    def test_single_atom_isotropic_convergence(self, sigma):
        """
        Test that coefficients are correct, such that the approximation using
        these coefficients can reasonably recreate the original isotropic
        atomic Gaussian.
        """

        frame = Atoms(
            positions=np.array([[0.0, 0.0, 0.0]]),
            numbers=[0],
        )
        frame.arrays["quaternions"] = np.array([[1.0, 0.0, 0.0, 0.0]])

        frame.arrays["c_diameter[1]"] = 2.0 * sigma * np.ones(1)
        frame.arrays["c_diameter[2]"] = 2.0 * sigma * np.ones(1)
        frame.arrays["c_diameter[3]"] = 2.0 * sigma * np.ones(1)
        frames = [frame]

        rgw = sigma + 0.5

        max_radials = range(10)
        errs = []

        r_mesh = torch.linspace(0.0, 5.0, 100, dtype=torch.float64)
        length_norm = (sigma**3 * (2.0 * torch.pi) ** 1.5) ** -1.0
        actual = length_norm * torch.exp(-(r_mesh**2) / (2.0 * sigma**2))

        for max_radial in max_radials:
            hypers = {
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

            representation = EllipsoidalDensityProjection(**hypers)

            descriptor_raw = representation.transform(frames, normalize=True)
            descriptor = mts.operations.sum_over_samples(
                descriptor_raw,
                sample_names="center",
            )

            def real_sphharm(m, l, theta, phi):
                m = int(m)
                l = int(l)

                if m < 0:
                    return (
                        torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
                        * (-1.0) ** m
                        * torch.imag(
                            torch.as_tensor(
                                sph_harm_y(l, abs(m), theta, phi),
                                dtype=torch.complex128,
                            )
                        )
                    )
                if m == 0:
                    return torch.real(
                        torch.as_tensor(
                            sph_harm_y(l, 0, theta, phi),
                            dtype=torch.complex128,
                        )
                    )

                return (
                    torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
                    * (-1.0) ** m
                    * torch.real(
                        torch.as_tensor(
                            sph_harm_y(l, m, theta, phi),
                            dtype=torch.complex128,
                        )
                    )
                )

            def evaluate_bases(r):
                bases = descriptor.copy()

                for key, block in bases.items():
                    l = int(key["angular_channel"])

                    for m_index, m_value in enumerate(
                        block.components[0]["spherical_component_m"]
                    ):
                        m = int(m_value)
                        ylm = real_sphharm(m, l, 0.0, 0.0)

                        basis_values = representation.radial_basis.get_basis(
                            r.reshape(1)
                        ).flatten()

                        for n_index, n_value in enumerate(block.properties["n"]):
                            n = int(n_value)
                            rnl = basis_values[n]
                            block.values[0, m_index, n_index] = ylm * rnl

                return bases

            approx = []
            for r in r_mesh:
                bases = evaluate_bases(r)
                approx.append(
                    torch.sum(bases.block(0).values * descriptor.block(0).values)
                )

            approx = torch.stack(approx)
            err = torch.mean((approx - actual) ** 2)
            errs.append(err)

        errs = torch.stack(errs)

        assert torch.all(errs[:-1] >= errs[1:])

        assert_allclose(
            approx.detach().cpu().numpy(),
            actual.detach().cpu().numpy(),
            atol=1e-3,
        )
