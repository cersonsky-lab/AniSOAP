from dataclasses import replace

import numpy as np

from anisoap.benchmarks.gb_audit import (
    angular_balance_residual,
    force_balance_residual,
    quaternion_matrices,
)
from anisoap.benchmarks.gb_dataset import GBRecord
from anisoap.benchmarks.gb_training import build_parser


def _record():
    positions = np.asarray([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    forces = np.asarray([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    # Orbital moment is [0, 0, -1], so stored space torques sum to [0,0,1].
    torques = np.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    quaternions = np.asarray([[0.0, 0.0, 0.0, 1.0]] * 2)
    return GBRecord(
        positions=positions,
        quaternions=quaternions,
        diameters=np.ones((2, 3)),
        energy=0.0,
        forces=forces,
        torques=torques,
        has_forces=True,
        has_torques=True,
        distance=1.0,
        source="random_rotations",
        metadata={},
    )


def test_force_and_angular_balance():
    record = _record()
    np.testing.assert_allclose(force_balance_residual(record), 0.0)
    np.testing.assert_allclose(
        angular_balance_residual(
            record,
            ordering="xyzw",
            convention="stored_space",
            orbital_sign=1,
        ),
        0.0,
        atol=1.0e-14,
    )

    minus_residual = angular_balance_residual(
        record,
        ordering="xyzw",
        convention="stored_space",
        orbital_sign=-1,
    )
    np.testing.assert_allclose(minus_residual, [0.0, 0.0, 2.0])


def test_invalid_orbital_sign_is_rejected():
    record = _record()
    with np.testing.assert_raises_regex(
        ValueError,
        "orbital_sign must be",
    ):
        angular_balance_residual(
            record,
            ordering="xyzw",
            convention="stored_space",
            orbital_sign=0,
        )


def test_quaternion_orderings_recover_identity():
    xyzw = quaternion_matrices(
        np.asarray([[0.0, 0.0, 0.0, 1.0]]),
        "xyzw",
    )
    wxyz = quaternion_matrices(
        np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        "wxyz",
    )
    np.testing.assert_allclose(xyzw[0], np.eye(3), atol=1.0e-14)
    np.testing.assert_allclose(wxyz[0], np.eye(3), atol=1.0e-14)


def test_parser_accepts_all_loss_ablations():
    for term in ("energy", "force", "torque"):
        args = build_parser().parse_args(
            [
                "--overfit-random",
                "8",
                "--disable-char-curves",
                "--loss-ablation",
                term,
            ]
        )
        assert args.loss_ablation == term
