import metatensor.torch as mts
from pathlib import Path
import torch
import pytest
import numpy as np
import os

from ase import Atoms
from ase.io import read
from numpy.testing import assert_allclose
from scipy.special import sph_harm_y

from anisoap.representations import EllipsoidalDensityProjection


class TestTorchCorrectness:
    def test_benzene_correctness_5frames(self):
        """
        Tests that the torch implementation matches the previous numpy anisoap implementation for the first five frames of benzenes
        """
        lmax = 9
        nmax = 6
        repo_root = Path(__file__).resolve().parents[1]
        frames = read(repo_root / "../notebooks" / "ellipsoids.xyz", ":5")

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

        calculator = EllipsoidalDensityProjection(**AniSOAP_HYPERS)
        x_anisoap_torch = calculator.power_spectrum(frames)

        x_anisoap_numpy = np.load(
            repo_root / "integration-tests/benzene_numpy_impl_5frames.npy"
        )
        assert_allclose(x_anisoap_torch, x_anisoap_numpy, rtol=0, atol=1e-2)
