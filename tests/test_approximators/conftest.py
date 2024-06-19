
import keras
import pytest

import bayesflow as bf


@pytest.fixture()
def summary_network():
    return None


@pytest.fixture()
def inference_network():
    network = keras.Sequential([
        keras.layers.Dense(10)
    ])
    network.compile(loss="mse")
    return network


@pytest.fixture()
def approximator(inference_network, summary_network):
    return bf.Approximator(
        inference_network=inference_network,
        summary_network=summary_network,
        inference_variables=[],
        inference_conditions=[],
        summary_variables=[],
        summary_conditions=[],
    )


@pytest.fixture()
def dataset():
    batch_size = 16
    batches_per_epoch = 4
    parameter_sets = batch_size * batches_per_epoch
    observations_per_parameter_set = 32

    mean = keras.random.normal(mean=0.0, stddev=0.1, shape=(parameter_sets, 2))
    std = keras.ops.exp(keras.random.normal(mean=0.0, stddev=0.1, shape=(parameter_sets, 2)))

    mean = keras.ops.repeat(mean[:, None], observations_per_parameter_set, 1)
    std = keras.ops.repeat(std[:, None], observations_per_parameter_set, 1)

    noise = keras.random.normal(shape=(parameter_sets, observations_per_parameter_set, 2))

    x = mean + std * noise

    data = dict(observables=dict(x=x), parameters=dict(mean=mean, std=std))

    return bf.datasets.OfflineDataset(data, batch_size=batch_size, batches_per_epoch=batches_per_epoch)
