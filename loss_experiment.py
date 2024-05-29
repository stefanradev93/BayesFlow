
import os
os.environ["KERAS_BACKEND"] = "torch"

import keras


class MyLayer(keras.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(None)

    def build(self, x1_shape, x2_shape, conditions_shape=None):
        self.dense.units = x2_shape[-1]
        self.dense.build(x1_shape)

    def call(self, x1, x2, conditions=None):
        return x2 + self.dense(x1)


layer = MyLayer()
x1 = keras.random.normal((128, 2))
x2 = keras.random.normal((128, 3))
layer(x1, x2)
