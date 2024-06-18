import math

import keras
import numpy as np
import pytest


@pytest.fixture()
def batch_size():
    return 32


@pytest.fixture()
def simulator():
    from bayesflow.experimental.simulators import SequentialSimulator

    def contexts():
        r = np.random.normal(0.1, 0.01)
        alpha = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)

        return dict(r=r, alpha=alpha)

    def parameters():
        theta = np.random.uniform(-1.0, 1.0, size=2)

        return dict(theta=theta)

    def observables(r, alpha, theta):
        x1 = -keras.ops.abs(theta[0] + theta[1]) / np.sqrt(2.0) + r * keras.ops.cos(alpha) + 0.25
        x2 = (-theta[0] + theta[1]) / np.sqrt(2.0) + r * keras.ops.sin(alpha)

        return dict(x=keras.ops.stack([x1, x2]))

    simulator = SequentialSimulator([contexts, parameters, observables])

    return simulator

    # class Simulator:
    #     def sample(self, batch_shape):
    #         r = keras.random.normal(shape=batch_shape + (1,), mean=0.1, stddev=0.01)
    #         alpha = keras.random.uniform(shape=batch_shape + (1,), minval=-0.5 * math.pi, maxval=0.5 * math.pi)
    #         theta = keras.random.uniform(shape=batch_shape + (2,), minval=-1.0, maxval=1.0)
    #
    #         x1 = -keras.ops.abs(theta[..., :1] + theta[..., 1:]) / keras.ops.sqrt(2.0) + r * keras.ops.cos(alpha) + 0.25
    #         x2 = (-theta[..., :1] + theta[..., 1:]) / keras.ops.sqrt(2.0) + r * keras.ops.sin(alpha)
    #
    #         x = keras.ops.concatenate([x1, x2], axis=-1)
    #
    #         return dict(r=r, alpha=alpha, theta=theta, x=x)
    #
    # return Simulator()


@pytest.fixture()
def train_dataset(simulator, batch_size):
    from bayesflow.experimental.datasets import OfflineDataset
    data = simulator.sample((16 * batch_size,))
    return OfflineDataset(data, workers=4, max_queue_size=16, batch_size=batch_size)


@pytest.fixture()
def validation_dataset(simulator, batch_size):
    from bayesflow.experimental.datasets import OfflineDataset
    data = simulator.sample((4 * batch_size,))
    return OfflineDataset(data, workers=4, max_queue_size=16, batch_size=batch_size)


@pytest.fixture()
def inference_network():
    from bayesflow.experimental.networks import CouplingFlow
    return CouplingFlow()


@pytest.fixture()
def approximator(inference_network):
    from bayesflow.experimental.backend_approximators import Approximator
    return Approximator(
        inference_network=inference_network,
        inference_variables=["theta"],
        inference_conditions=["x", "r", "alpha"],
    )
