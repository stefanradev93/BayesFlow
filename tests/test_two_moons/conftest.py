import math

import keras
import pytest

import bayesflow.experimental as bf


@pytest.fixture()
def batch_size():
    return 32


@pytest.fixture()
def simulator():
    class Simulator:
        def sample(self, batch_shape):
            r = keras.random.normal(shape=batch_shape + (1,), mean=0.1, stddev=0.01)
            alpha = keras.random.uniform(shape=batch_shape + (1,), minval=-0.5 * math.pi, maxval=0.5 * math.pi)
            theta = keras.random.uniform(shape=batch_shape + (2,), minval=-1.0, maxval=1.0)

            x1 = -keras.ops.abs(theta[0] + theta[1]) / keras.ops.sqrt(2.0) + r * keras.ops.cos(alpha) + 0.25
            x2 = (-theta[0] + theta[1]) / keras.ops.sqrt(2.0) + r * keras.ops.sin(alpha)

            x = keras.ops.stack([x1, x2], axis=-1)

            return dict(r=r, alpha=alpha, theta=theta, x=x)

    return Simulator()


@pytest.fixture()
def dataset(joint_distribution):
    return bf.datasets.OnlineDataset(joint_distribution, workers=4, use_multiprocessing=True, max_queue_size=16, batch_size=16)


@pytest.fixture()
def inference_network():
    return bf.networks.CouplingFlow()


@pytest.fixture()
def approximator(inference_network):
    return bf.Approximator(
        inference_network=inference_network,
        inference_variables=["theta"],
        inference_conditions=["x", "r", "alpha"],
        summary_network=None,
        summary_variables=[],
        summary_conditions=[],
    )
