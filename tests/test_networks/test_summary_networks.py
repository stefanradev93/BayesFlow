
import keras
import numpy as np
import pytest

from tests.utils import assert_layers_equal


@pytest.mark.parametrize("automatic", [True, False])
def test_build(automatic, summary_network, random_set):
    assert summary_network.built is False

    if automatic:
        summary_network(random_set)
    else:
        summary_network.build(keras.ops.shape(random_set))

    assert summary_network.built is True

    # check the model has variables
    assert summary_network.variables, "Model has no variables."


def test_variable_batch_size(summary_network, random_set):
    # build with one batch size
    summary_network.build(keras.ops.shape(random_set))

    # run with another batch size
    for _ in range(10):
        batch_size = np.random.randint(1, 10)
        new_input = keras.ops.zeros((batch_size,) + keras.ops.shape(random_set)[1:])
        summary_network(new_input)


def test_variable_set_size(summary_network, random_set):
    # build with one set size
    summary_network.build(keras.ops.shape(random_set))

    # run with another set size
    for _ in range(10):
        batch_size = keras.ops.shape(random_set)[0]
        set_size = np.random.randint(1, 10)
        new_input = keras.ops.zeros((batch_size, set_size, keras.ops.shape(random_set)[2]))
        summary_network(new_input)


def test_serialize_deserialize(tmp_path, summary_network, random_set):
    summary_network.build(keras.ops.shape(random_set))

    keras.saving.save_model(summary_network, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(summary_network, loaded)
