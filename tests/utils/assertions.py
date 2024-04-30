
import keras


def assert_models_equal(model1: keras.Model, model2: keras.Model):
    for v1, v2 in zip(model1.variables, model2.variables):
        assert keras.ops.all(keras.ops.isclose(v1, v2))
