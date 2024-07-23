import keras

from .functional import maximum_mean_discrepancy


class MaximumMeanDiscrepancy(keras.metrics.MeanMetricWrapper):
    def __init__(self, name="maximum_mean_discrepancy", dtype=None, **kwargs):
        def fn(y_true, y_pred):
            mmd = maximum_mean_discrepancy(y_true, y_pred, **kwargs)
            return keras.ops.mean(mmd, axis=1)

        super().__init__(fn, name=name, dtype=dtype)
