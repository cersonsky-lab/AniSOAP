from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from anisoap.benchmarks.gb_dataset import (
    characteristic_curve_key,
    load_gb_dataset,
    split_characteristic_curves,
    split_random_rotations,
)


def _frame(
    *,
    distance,
    energy,
    with_forces,
    with_torques,
    curve_key=None,
):
    frame = Atoms(
        numbers=[1, 1],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, distance]],
    )
    frame.arrays["quaternions"] = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    for axis, value in enumerate((1.0, 1.2, 1.5), start=1):
        frame.arrays[f"c_diameter[{axis}]"] = np.asarray([value, value])

    results = {"energy": float(energy)}
    if with_forces:
        results["forces"] = np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
        )
    frame.calc = SinglePointCalculator(frame, **results)

    if with_torques:
        frame.arrays["torques"] = np.asarray(
            [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]
        )
        frame.info["separation_distance"] = float(distance)

    if curve_key is not None:
        phi_x, phi_y, phi_z, theta = curve_key
        frame.info.update(
            {
                "phi_x": phi_x,
                "phi_y": phi_y,
                "phi_z": phi_z,
                "theta": theta,
                "d": float(distance),
                "h12": float(distance - 2.0),
            }
        )

    return frame


def _write(path: Path, frames):
    write(path, frames, format="extxyz")


def test_load_random_rotations(tmp_path):
    path = tmp_path / "random_rotations.xyz"
    frames = [
        _frame(
            distance=2.0 + 0.1 * index,
            energy=-0.5 + 0.01 * index,
            with_forces=True,
            with_torques=True,
        )
        for index in range(5)
    ]
    _write(path, frames)

    dataset = load_gb_dataset(path, source="random_rotations")

    assert len(dataset) == 5
    assert dataset.source == "random_rotations"

    record = dataset[0]
    assert record.positions.shape == (2, 3)
    assert record.quaternions.shape == (2, 4)
    assert record.diameters.shape == (2, 3)
    assert record.forces.shape == (2, 3)
    assert record.torques.shape == (2, 3)
    assert record.has_forces
    assert record.has_torques


def test_load_characteristic_curves(tmp_path):
    path = tmp_path / "char_curves.xyz"
    frames = [
        _frame(
            distance=2.0 + 0.1 * index,
            energy=-0.8 + 0.02 * index,
            with_forces=False,
            with_torques=False,
            curve_key=(0.0, 0.0, float(index % 2), 0.5),
        )
        for index in range(6)
    ]
    _write(path, frames)

    dataset = load_gb_dataset(path, source="char_curves")

    assert len(dataset) == 6
    record = dataset[0]
    assert record.forces is None
    assert record.torques is None
    assert not record.has_forces
    assert not record.has_torques
    assert characteristic_curve_key(record) == (0.0, 0.0, 0.0, 0.5)


def test_random_rotation_split_is_disjoint_and_spans_distances(tmp_path):
    path = tmp_path / "random_rotations.xyz"
    frames = [
        _frame(
            distance=1.8 + 0.02 * index,
            energy=float(index),
            with_forces=True,
            with_torques=True,
        )
        for index in range(60)
    ]
    _write(path, frames)
    dataset = load_gb_dataset(path, source="random_rotations")

    split = split_random_rotations(
        dataset,
        fractions=(0.6, 0.2, 0.2),
        distance_bins=6,
        seed=7,
    )
    split.validate(len(dataset))

    distances = np.asarray([record.distance for record in dataset.records])
    for indices in (split.train, split.validation, split.test):
        assert len(indices) > 0
        assert distances[indices].min() < distances.mean()
        assert distances[indices].max() > distances.mean()


def test_characteristic_curve_split_keeps_curves_together(tmp_path):
    path = tmp_path / "char_curves.xyz"
    frames = []

    keys = [
        (0.0, 0.0, float(index), 0.5)
        for index in range(10)
    ]
    for key in keys:
        for distance in (2.0, 2.2, 2.4, 2.6):
            frames.append(
                _frame(
                    distance=distance,
                    energy=distance,
                    with_forces=False,
                    with_torques=False,
                    curve_key=key,
                )
            )

    _write(path, frames)
    dataset = load_gb_dataset(path, source="char_curves")
    split = split_characteristic_curves(
        dataset,
        fractions=(0.6, 0.2, 0.2),
        seed=9,
    )
    split.validate(len(dataset))

    memberships = {}
    for name, indices in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        for index in indices:
            key = characteristic_curve_key(dataset[int(index)])
            memberships.setdefault(key, set()).add(name)

    assert all(len(names) == 1 for names in memberships.values())
    assert len(memberships) == len(keys)


def test_missing_random_rotation_torque_is_rejected(tmp_path):
    path = tmp_path / "bad.xyz"
    frame = _frame(
        distance=2.0,
        energy=-0.5,
        with_forces=True,
        with_torques=False,
    )
    _write(path, [frame])

    with pytest.raises(ValueError, match="torques"):
        load_gb_dataset(path, source="random_rotations")


def test_max_frames(tmp_path):
    path = tmp_path / "random_rotations.xyz"
    frames = [
        _frame(
            distance=2.0 + index * 0.1,
            energy=float(index),
            with_forces=True,
            with_torques=True,
        )
        for index in range(5)
    ]
    _write(path, frames)

    dataset = load_gb_dataset(
        path,
        source="random_rotations",
        max_frames=2,
    )
    assert len(dataset) == 2
