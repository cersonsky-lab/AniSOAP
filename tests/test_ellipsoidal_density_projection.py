import builtins

import ase
import equistore
import numpy as np
import pytest

from anisoap.representations import EllipsoidalDensityProjection
from numpy.testing import assert_allclose


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
TEST_MATRIX_FRAME = TEST_SINGLE_FRAME.copy()
TEST_MATRIX_FRAME.arrays["matrix"] = [np.eye(3)]

TEST_FRAMES = [
    [TEST_SINGLE_FRAME],
    [TEST_QUAT_FRAME],
    [TEST_MATRIX_FRAME],
    [TEST_SINGLE_FRAME, TEST_QUAT_FRAME, TEST_MATRIX_FRAME],
]


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

    @pytest.mark.parametrize("frames", TEST_FRAMES)
    def test_frames_show_progress(self, frames):
        EllipsoidalDensityProjection(**DEFAULT_HYPERS).transform(
            frames, show_progress=True
        )

    @pytest.mark.parametrize("frames", TEST_FRAMES)
    def test_frames_matrix_rotation(self, frames):
        EllipsoidalDensityProjection(
            rotation_key="matrix", rotation_type="matrix", **DEFAULT_HYPERS
        ).transform(frames, show_progress=True)

    @pytest.mark.parametrize("frames", TEST_FRAMES)
    def test_frames_normalization_condition(self, frames):
        edp = EllipsoidalDensityProjection(
            rotation_key="matrix", rotation_type="matrix", **DEFAULT_HYPERS
        )
        rep_unnormalized = edp.transform(frames, normalize=False)
        rep_normalized_1 = edp.transform(frames, normalize=True)
        rep_normalized_2 = edp.radial_basis.orthonormalize_basis(rep_unnormalized)

        # Would do this, but failing GitHub CI for older versions of python (possibly b/c it's
        # building an older version of equistore)?
        # assert equistore.allclose(rep_normalized_1, rep_normalized_2)
        for i in range(len(rep_unnormalized.blocks())):
            block_norm_1 = rep_normalized_1.block(i)
            block_norm_2 = rep_normalized_2.block(i)
            assert_allclose(
                block_norm_1.values, block_norm_2.values, rtol=1e-10, atol=1e-10
            )


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
        [
            {**DEFAULT_HYPERS, "radial_gaussian_width": 9},
            ValueError,
            "radial_gaussian_width is set as an integer, which could cause overflow errors. Pass in float."
        ]
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
