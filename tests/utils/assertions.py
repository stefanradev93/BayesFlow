
import keras


def assert_models_equal(model1: keras.Model, model2: keras.Model):
    assert isinstance(model1, keras.Model)
    assert isinstance(model2, keras.Model)

    for layer1, layer2 in zip(model1.layers, model2.layers):
        if isinstance(layer1, keras.Model):
            assert_models_equal(layer1, layer2)
        else:
            assert_layers_equal(layer1, layer2)


def assert_layers_equal(layer1: keras.Layer, layer2: keras.Layer):
    assert len(layer1.variables) == len(layer2.variables), f"Layers {layer1.name} and {layer2.name} have a different number of variables ({len(layer1.variables)}, {len(layer2.variables)})."
    assert len(layer1.variables) > 0, f"Layers {layer1.name} and {layer2.name} have no variables."
    for v1, v2 in zip(layer1.variables, layer2.variables):
        v1 = keras.ops.convert_to_numpy(v1)
        v2 = keras.ops.convert_to_numpy(v2)
        assert keras.ops.all(keras.ops.isclose(v1, v2)), f"Variables for {layer1.name} and {layer2.name} are not equal: {v1} != {v2}"
