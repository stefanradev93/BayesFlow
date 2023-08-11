from unittest.mock import Mock

import pytest
import numpy as np
from bayesflow import computational_utilities
from bayesflow.exceptions import ArgumentError, ShapeError
from bayesflow.trainers import Trainer
import tensorflow as tf


@pytest.mark.parametrize("x_true, x_pred, output",
                         [
                             (0, [0, 0, 0], 0),
                             (0, np.array([0, 0, 0]), 0),
                             (np.array(0), np.array(0), 0),
                             (0, [0], 0),
                             (10, [10, 10, 10], 0),
                             (10, [8, 12], 4)
                         ],
                         )
def test_mean_squared_error(x_true, x_pred, output):
    assert computational_utilities.mean_squared_error(x_true=x_true, x_pred=x_pred) == pytest.approx(output)


@pytest.mark.parametrize("x_true, x_pred, output",
                         [
                             (10, [8, 12], 2),
                             (0, [0, 0, 0], 0)
                         ]
                         )
def test_root_mean_squared_error(x_true, x_pred, output):
    assert computational_utilities.root_mean_squared_error(x_true=x_true, x_pred=x_pred) == pytest.approx(output)


@pytest.mark.parametrize("x_true, x_pred, inner_error_fun, outer_aggregation_fun, output",
                         [
                             # test case 1
                             ([0, 0, 0],
                              np.array([
                                  [0, 0, -1, 4, 0],
                                  [0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1]]),
                              computational_utilities.mean_squared_error,
                              np.median,
                              1.0
                              ),
                             # test case 2
                             ([0, 0, 1],
                              np.array([
                                  [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1]]),
                              computational_utilities.root_mean_squared_error,
                              np.mean,
                              0.0
                              ),
                             # test case 3
                             ([0, 0, 1],
                              np.array([
                                  [4, 0, 0, 0],
                                  [1, 1, -1, -1],
                                  [4, -2, 4, -2]]),
                              computational_utilities.root_mean_squared_error,
                              np.mean,
                              2.0
                              ),
                             # test case 4
                             ([0, 0, 1],
                              np.array([
                                  [4, 0, 0, 0],
                                  [1, 1, -1, -1],
                                  [4, -2, 4, -2]]),
                              computational_utilities.mean_squared_error,
                              np.mean,
                              np.mean([4, 1, 9])
                              ),
                             # test case 5
                             (
                                     np.array([[0, 1], [10, 10]]),
                                     np.array([
                                         [[1, 1], [0, 0]],
                                         [[11, 11], [9, 9]]
                                     ]),
                                     computational_utilities.mean_squared_error,
                                     np.mean,
                                     np.mean([0.5, 1])
                             )
                         ])
def test_aggregated_error(x_true, x_pred, inner_error_fun, outer_aggregation_fun, output):
    aggregated_error_result = computational_utilities.aggregated_error(
        x_true=x_true,
        x_pred=x_pred,
        inner_error_fun=inner_error_fun,
        outer_aggregation_fun=outer_aggregation_fun
    )
    assert aggregated_error_result == pytest.approx(output)


def test_c2st_shape_error():
    source_samples = np.random.random(size=(5, 2))
    target_samples = np.random.random(size=(5, 3))
    with pytest.raises(ShapeError):
        computational_utilities.c2st(source_samples, target_samples)


@pytest.mark.parametrize(
    "source_samples, target_samples",
    [
        (np.random.random((5, 2)), np.random.random((5, 2))),
        (np.random.random((10, 2)), np.random.random((5, 2))),
        (tf.constant(np.random.random((5, 2))), tf.constant(np.random.random((5, 2))))
    ]
)
def test_c2st(source_samples, target_samples):
    c2st_score = computational_utilities.c2st(source_samples, target_samples)
    assert 0.0 <= c2st_score <= 1.0


@pytest.mark.parametrize(
    "n_folds, scoring, normalize, seed, hidden_units_per_dim",
    [
        (3, "accuracy", False, 42, 5),
        (7, "f1", True, 12, 10)
    ]
)
def test_c2st_params(n_folds, scoring, normalize, seed, hidden_units_per_dim):
    source_samples = np.random.random((5, 2))
    target_samples = np.random.random((10, 2))
    _ = computational_utilities.c2st(
        source_samples=source_samples,
        target_samples=target_samples,
        n_folds=n_folds,
        scoring=scoring,
        normalize=normalize,
        seed=seed,
        hidden_units_per_dim=hidden_units_per_dim
    )
