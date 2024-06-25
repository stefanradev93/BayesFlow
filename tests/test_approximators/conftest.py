import keras
import pytest


# @pytest.fixture()
# def batch_size():
#     return 32


@pytest.fixture()
def summary_network():
    return None


@pytest.fixture()
def inference_network():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow()


@pytest.fixture()
def approximator(inference_network, summary_network):
    from bayesflow import Approximator

    return Approximator(
        inference_network=inference_network,
        summary_network=summary_network,
        inference_variables=["mean", "std"],
        inference_conditions=["x"],
    )


@pytest.fixture()
def train_dataset():
    # TODO
    return None


@pytest.fixture()
def validation_dataset():
    # TODO
    return None


@pytest.fixture()
def dataset():
    # TODO: Parameterize over num_batches
    # TODO: Write as simulator for train_dataset and validation_dataset
    from bayesflow import OfflineDataset

    batch_size = 16
    num_batches = 4
    parameter_sets = batch_size * num_batches
    observations_per_parameter_set = 32

    mean = keras.random.normal(mean=0.0, stddev=0.1, shape=(parameter_sets, 2))
    std = keras.ops.exp(keras.random.normal(mean=0.0, stddev=0.1, shape=(parameter_sets, 2)))

    mean = keras.ops.repeat(mean[:, None], observations_per_parameter_set, 1)
    std = keras.ops.repeat(std[:, None], observations_per_parameter_set, 1)

    noise = keras.random.normal(shape=(parameter_sets, observations_per_parameter_set, 2))

    x = mean + std * noise

    data = dict(mean=mean, std=std, x=x)

    return OfflineDataset(data, workers=1, max_queue_size=10, batch_size=batch_size)
