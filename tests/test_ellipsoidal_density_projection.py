import builtins

import ase
import numpy as np
import pytest

from anisoap.representations import EllipsoidalDensityProjection


def add_default_params(frame):
    frame.arrays["quaternion"] = [[1, 0, 0, 0] for _ in frame]
    frame.arrays["c_diameter[1]"] = [1 for _ in frame]
    frame.arrays["c_diameter[2]"] = [1 for _ in frame]
    frame.arrays["c_diameter[3]"] = [2 for _ in frame]

    return frame


TEST_SINGLE_FRAME = add_default_params(
    ase.Atoms(symbols=["X"], positions=np.zeros((1, 3)), cell=(10, 10, 10), pbc=False)
)
TEST_QUAT_FRAME = add_default_params(
    ase.Atoms(
        symbols=["X", "O"], positions=[np.zeros(3), np.ones(3)], cell=(10, 10, 10)
    )
)

TEST_FRAMES = [[TEST_SINGLE_FRAME], [TEST_QUAT_FRAME]]


DEFAULT_HYPERS = {
    "max_angular": 10,
    "radial_basis_name": "gto",
    "radial_gaussian_width": 5.0,
    "cutoff_radius": 1.0,
}


class TestEllipsoidalDensityProjection:
    """
    Class for testing if the EDP can run as-expected on certain things
    """

    @pytest.mark.parametrize("frames", TEST_FRAMES)
    def test_frames(self, frames):
        EllipsoidalDensityProjection(**DEFAULT_HYPERS).transform(frames)


class TestBadInputs:
    """
    Class for testing if EDP fails correctly with bad hypers
    """

    test_hypers = [
        [
            {**DEFAULT_HYPERS, "compute_gradients": True},
            NotImplementedError,
            "Sorry! Gradients have not yet been implemented",
        ],
        [
            {
                **{k: v for k, v in DEFAULT_HYPERS.items() if k != "radial_basis_name"},
                "radial_basis_name": "nonsense",
            },
            NotImplementedError,
            "nonsense is not an implemented basis" ". Try 'monomial' or 'gto'",
        ],
        [
            {
                **{k: v for k, v in DEFAULT_HYPERS.items() if k != "radial_basis_name"},
                "radial_basis_name": "monomial",
            },
            ValueError,
            "Gaussian width can only be provided with GTO basis",
        ],
        [
            {
                **{
                    k: v
                    for k, v in DEFAULT_HYPERS.items()
                    if k != "radial_gaussian_width"
                }
            },
            ValueError,
            "Gaussian width must be provided with GTO basis",
        ],
        [
            {**DEFAULT_HYPERS, "rotation_type": "quaternions"},
            ValueError,
            "We have only implemented transforming quaternions (`quaternion`) and rotation matrices (`matrix`).",
        ],
    ]

    @pytest.mark.parametrize("hypers,error_type,expected_message", test_hypers)
    def test_hypers(self, hypers, error_type, expected_message):
        with pytest.raises(error_type) as cm:
            EllipsoidalDensityProjection(**hypers).transform(TEST_SINGLE_FRAME)
            assert cm.message == expected_message

    def test_no_rotations(self):
        frame = TEST_SINGLE_FRAME.copy()
        _ = frame.arrays.pop("quaternion")
        with pytest.warns() as cm:
            EllipsoidalDensityProjection(**DEFAULT_HYPERS).transform([frame])
            str(
                cm
            ) == f"Frame 0 does not have rotations stored, this may cause errors down the line."
