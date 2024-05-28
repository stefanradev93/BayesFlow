
import keras
import pytest

from tests.utils import assert_layers_equal


def test_manual_build(network):
    network.build((None, 2))

    assert network.built

    # correct shape
    test_input = keras.ops.zeros((128, 2))
    network(test_input)

    # second build
    network.build((None, 2))

    with pytest.raises(ValueError):
        # wrong shape
        network.build((None, 3))

    test_input = keras.ops.zeros((128, 3))
    with pytest.raises(ValueError):
        # wrong input shape
        network(test_input)


def test_automatic_build(network):
    test_input = keras.ops.zeros((128, 2))

    # auto-builds the network
    network(test_input)

    assert network.built

    # test second build
    network.build((None, 2))

    with pytest.raises(ValueError):
        # wrong shape
        network.build((None, 3))

    test_input = keras.ops.zeros((128, 3))
    with pytest.raises(ValueError):
        # wrong input shape
        network(test_input)


@pytest.mark.skip(reason="TODO: Skip if network is not invertible")
def test_forward_inverse(network):
    # TODO: skip if network is not invertible
    original_data = keras.random.normal((128, 2))

    latent = network(original_data)
    data = network(latent, inverse=True)

    assert keras.ops.is_tensor(latent)
    assert keras.ops.is_tensor(data)

    assert keras.ops.shape(latent) == keras.ops.shape(original_data)
    assert keras.ops.shape(data) == keras.ops.shape(original_data)

    assert keras.ops.all(keras.ops.isclose(data, original_data))


def test_serialize_deserialize(tmp_path, network):
    network.build((None, 2))
    keras.saving.save_model(network, tmp_path / "model.keras")
    loaded_network = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(network, loaded_network)
