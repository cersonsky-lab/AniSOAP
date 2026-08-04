import numpy as np
from scipy.spatial.transform import Rotation

from anisoap.benchmarks.gb_training import (
    _body_to_space_vectors,
    _space_to_body_vectors,
    build_parser,
)


def test_body_space_vector_round_trip_wxyz():
    rotation_sb = Rotation.from_euler("xyz", [0.2, -0.4, 0.7])
    xyzw = rotation_sb.as_quat()
    wxyz = xyzw[[3, 0, 1, 2]]
    body = np.asarray([[0.3, -1.2, 2.0]])

    space = _body_to_space_vectors(
        body,
        wxyz[None, :],
        ordering="wxyz",
    )
    recovered = _space_to_body_vectors(
        space,
        wxyz[None, :],
        ordering="wxyz",
    )

    np.testing.assert_allclose(recovered, body, atol=1.0e-14)


def test_body_to_space_uses_transpose_of_space_to_body():
    rotation_sb = Rotation.from_euler("z", np.pi / 2)
    xyzw = rotation_sb.as_quat()
    wxyz = xyzw[[3, 0, 1, 2]]
    body = np.asarray([[1.0, 0.0, 0.0]])

    space = _body_to_space_vectors(
        body,
        wxyz[None, :],
        ordering="wxyz",
    )
    expected = rotation_sb.as_matrix().T @ body[0]
    np.testing.assert_allclose(space[0], expected, atol=1.0e-14)


def test_parser_defaults_to_body_torque_labels():
    args = build_parser().parse_args([])
    assert args.torque_target_frame == "body"
