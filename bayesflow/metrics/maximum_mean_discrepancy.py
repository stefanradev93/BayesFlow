import keras

from .functional import maximum_mean_discrepancy


class MaximumMeanDiscrepancy(keras.metrics.MeanMetricWrapper):
    def __init__(self, name="maximum_mean_discrepancy", dtype=None, **kwargs):
        super().__init__(maximum_mean_discrepancy, name=name, dtype=dtype)
