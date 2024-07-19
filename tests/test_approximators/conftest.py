import pytest


@pytest.fixture()
def batch_size():
    return 8


@pytest.fixture()
def summary_network():
    return None


@pytest.fixture()
def inference_network():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow()


@pytest.fixture()
def approximator(inference_network, summary_network):
    from bayesflow import ContinuousApproximator

    return ContinuousApproximator(
        inference_network=inference_network,
        summary_network=summary_network,
        inference_variables=["mean", "std"],
        inference_conditions=["x"],
    )


@pytest.fixture()
def simulator():
    from tests.utils.normal_simulator import NormalSimulator

    return NormalSimulator()


@pytest.fixture()
def train_dataset(simulator, batch_size):
    from bayesflow import OfflineDataset

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))
    return OfflineDataset(data, workers=4, max_queue_size=num_batches, batch_size=batch_size)


@pytest.fixture()
def validation_dataset(simulator, batch_size):
    from bayesflow import OfflineDataset

    num_batches = 2
    data = simulator.sample((num_batches * batch_size,))
    return OfflineDataset(data, workers=4, max_queue_size=num_batches, batch_size=batch_size)
