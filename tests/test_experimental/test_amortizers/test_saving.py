
import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import keras.saving

import pytest

import bayesflow.experimental as bf

from tests.utils import *


@pytest.fixture(scope="module")
def summary_network():
    # TODO: modularize over data shape
    # TODO: add no summary network case
    return keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(2),
        keras.layers.Lambda(lambda x: keras.ops.mean(x, axis=1, keepdims=True))
    ])


@pytest.fixture(scope="module")
def inference_network():
    # TODO: modularize over data shape
    class AffineSubnet(keras.Layer):
        def __init__(self, in_features, out_features, **kwargs):
            super().__init__(**kwargs)
            self.network = keras.Sequential([
                keras.layers.Input(shape=(in_features,)),
                keras.layers.Dense(out_features),
            ])

        def call(self, x):
            scale, shift = keras.ops.split(self.network(x), 2, axis=1)
            return dict(scale=scale, shift=shift)

    return bf.networks.CouplingFlow.uniform(
        subnet_constructor=AffineSubnet,
        features=2,
        conditions=0,
        layers=1,
        transform="affine",
        base_distribution="normal",
    )


@pytest.fixture(scope="module")
def model(inference_network, summary_network):
    return bf.Amortizer(inference_network, summary_network)


def test_save_and_load(tmp_path, model):
    path = tmp_path / "model.keras"
    model.save(path)
    loaded_model = keras.saving.load_model(path)

    assert_models_equal(model, loaded_model)
