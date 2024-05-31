
import keras


def assert_layers_equal(layer1: keras.Layer, layer2: keras.Layer):
    assert layer1.variables, "Layer has no variables."
    for v1, v2 in zip(layer1.variables, layer2.variables):
        assert keras.ops.all(keras.ops.isclose(v1, v2)), f"Variables not equal: {v1} != {v2}"
