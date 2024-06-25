import keras

from . import functional


class MaximumMeanDiscrepancy(keras.Metric):
    def __init__(self, kernel: str = "gaussian", kernel_kwargs: dict = None):
        super().__init__(name="maximum_mean_discrepancy")
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs or {}

        self.values = []

    def update_state(self, y_true, y_pred):
        mmd = functional.maximum_mean_discrepancy(y_true, y_pred, kernel=self.kernel, **self.kernel_kwargs)
        self.values.append(mmd)

    def result(self):
        return keras.ops.mean(keras.ops.concatenate(self.values, axis=0))
