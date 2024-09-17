from functools import partial
import keras


from .functional import maximum_mean_discrepancy


class MaximumMeanDiscrepancy(keras.metrics.MeanMetricWrapper):
    def __init__(self, name="maximum_mean_discrepancy", dtype=None, **kwargs):
        fn = partial(maximum_mean_discrepancy, **kwargs)
        super().__init__(fn, name=name, dtype=dtype)
