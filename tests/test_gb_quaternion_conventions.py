import numpy as np
from scipy.spatial.transform import Rotation

from anisoap.benchmarks.gb_training import (
    _space_to_body_as_anisoap_quaternion,
    build_parser,
)


def test_wxyz_space_to_body_is_reordered_to_xyzw():
    angle = np.pi / 3
    xyzw = Rotation.from_euler("z", angle).as_quat()
    wxyz = xyzw[[3, 0, 1, 2]]

    converted = _space_to_body_as_anisoap_quaternion(
        wxyz[None, :],
        ordering="wxyz",
        matrix_direction="space_to_body",
    )[0]

    expected = Rotation.from_quat(xyzw).as_matrix()
    actual = Rotation.from_quat(converted).as_matrix()
    np.testing.assert_allclose(actual, expected, atol=1.0e-14)


def test_body_to_space_option_transposes_stored_matrix():
    xyzw = Rotation.from_euler("xyz", [0.2, -0.4, 0.7]).as_quat()
    converted = _space_to_body_as_anisoap_quaternion(
        xyzw[None, :],
        ordering="xyzw",
        matrix_direction="body_to_space",
    )[0]

    stored = Rotation.from_quat(xyzw).as_matrix()
    actual = Rotation.from_quat(converted).as_matrix()
    np.testing.assert_allclose(actual, stored.T, atol=1.0e-14)


def test_parser_defaults_to_wxyz_space_to_body():
    args = build_parser().parse_args([])
    assert args.quaternion_order == "wxyz"
    assert args.quaternion_matrix_direction == "space_to_body"
