import json

import numpy as np

from anisoap.benchmarks.gb_training import (
    regression_metrics,
    save_learning_curve,
    save_parity_plot,
)


def test_regression_metrics_exact_prediction():
    target = np.asarray([-1.0, 0.0, 2.0])
    metrics = regression_metrics(target, target.copy())

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_regression_metrics_known_residual():
    target = np.asarray([0.0, 1.0])
    prediction = np.asarray([1.0, 1.0])
    metrics = regression_metrics(target, prediction)

    np.testing.assert_allclose(metrics["mae"], 0.5)
    np.testing.assert_allclose(metrics["rmse"], np.sqrt(0.5))
    np.testing.assert_allclose(metrics["r2"], -1.0)


def test_save_parity_plot(tmp_path):
    path = tmp_path / "parity.png"
    save_parity_plot(
        np.asarray([-1.0, 0.0, 1.0]),
        np.asarray([-0.8, 0.1, 0.9]),
        path,
        title="Test parity",
        label="energy",
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_save_learning_curve(tmp_path):
    path = tmp_path / "learning.png"
    save_learning_curve(
        [
            {"epoch": 1, "train_loss": 2.0, "validation_loss": 2.5},
            {"epoch": 2, "train_loss": 1.0, "validation_loss": 1.5},
        ],
        path,
    )

    assert path.exists()
    assert path.stat().st_size > 0
