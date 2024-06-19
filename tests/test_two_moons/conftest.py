import pytest


@pytest.fixture()
def approximator(inference_network):
    from bayesflow import Approximator
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
    from bayesflow.networks import CouplingFlow
    return CouplingFlow()


@pytest.fixture()
def random_samples(batch_size, simulator):
    return simulator.sample((batch_size,))


@pytest.fixture()
def simulator():
    from bayesflow.simulators import TwoMoonsSimulator
    return TwoMoonsSimulator()


@pytest.fixture()
def train_dataset(simulator, batch_size):
    from bayesflow import OfflineDataset
    num_batches = 16
    data = simulator.sample((num_batches * batch_size,))
    return OfflineDataset(data, workers=4, max_queue_size=num_batches, batch_size=batch_size)


@pytest.fixture()
def validation_dataset(simulator, batch_size):
    from bayesflow import OfflineDataset
    num_batches = 4
    data = simulator.sample((4 * batch_size,))
    return OfflineDataset(data, workers=4, max_queue_size=num_batches, batch_size=batch_size)
