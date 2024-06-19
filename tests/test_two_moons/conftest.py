import keras
import numpy as np
import pytest


@pytest.fixture()
def approximator(inference_network):
    from bayesflow.experimental import Approximator
    return Approximator(
        inference_network=inference_network,
        inference_variables=["theta"],
        inference_conditions=["x", "r", "alpha"],
    )


@pytest.fixture()
def batch_size():
    return 32


@pytest.fixture()
def inference_network():
    from bayesflow.experimental.networks import CouplingFlow
    return CouplingFlow()


@pytest.fixture()
def random_samples(batch_size, simulator):
    return simulator.sample((batch_size,))


@pytest.fixture()
def simulator():
    from bayesflow.experimental.simulators import SequentialSimulator

    def contexts():
        r = np.random.normal(0.1, 0.01)
        alpha = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)

        return dict(r=r, alpha=alpha)

    def parameters():
        theta = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)

        return dict(theta=theta)

    def observables(r, alpha, theta):
        x1 = -keras.ops.abs(theta[0] + theta[1]) / np.sqrt(2.0) + r * keras.ops.cos(alpha) + 0.25
        x2 = (-theta[0] + theta[1]) / np.sqrt(2.0) + r * keras.ops.sin(alpha)

        return dict(x=keras.ops.stack([x1, x2]))

    simulator = SequentialSimulator([contexts, parameters, observables])

    return simulator


@pytest.fixture()
def train_dataset(simulator, batch_size):
    from bayesflow.experimental import OfflineDataset
    num_batches = 16
    data = simulator.sample((num_batches * batch_size,))
    return OfflineDataset(data, workers=4, max_queue_size=num_batches, batch_size=batch_size)


@pytest.fixture()
def validation_dataset(simulator, batch_size):
    from bayesflow.experimental import OfflineDataset
    num_batches = 4
    data = simulator.sample((4 * batch_size,))
    return OfflineDataset(data, workers=4, max_queue_size=num_batches, batch_size=batch_size)
