from functools import partial
import keras


from .functional import root_mean_squared_error


class RootMeanSquaredError(keras.metrics.MeanMetricWrapper):
    def __init__(self, name="root_mean_squared_error", dtype=None, **kwargs):
        fn = partial(root_mean_squared_error, **kwargs)
        super().__init__(fn, name=name, dtype=dtype)
