
import keras
import pytest

import bayesflow.experimental as bf


@pytest.fixture()
def inference_network():
    # TODO: use actual inference network? (maybe too slow)
    return keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(2),
    ])


@pytest.fixture()
def summary_network():
    # TODO: add case for summary network
    return None


@pytest.fixture()
def amortizer(inference_network, summary_network):
    return bf.Amortizer(inference_network, summary_network)


@pytest.fixture()
def dataset():
    # TODO: construct a dummy dataset
    raise NotImplementedError
