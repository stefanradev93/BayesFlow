from unittest.mock import Mock

import pytest
import numpy as np
from bayesflow import computational_utilities
from bayesflow.exceptions import ArgumentError
from bayesflow.trainers import Trainer


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
