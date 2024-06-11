
import keras

from .approximator import Approximator


class JointApproximator(keras.Model):
    def __init__(self, **approximators: Approximator):
        super().__init__()
        self.approximators = approximators

    def build(self, input_shape):
        for approximator in self.approximators.values():
            approximator.build(input_shape)

    def call(self, *args, **kwargs):
        return {name: approximator(*args, **kwargs) for name, approximator in self.approximators.items()}

    def compute_loss(self, *args, **kwargs):
        losses = {name: amortizer.compute_loss(*args, **kwargs) for name, amortizer in self.approximators.items()}
        return keras.ops.mean(losses.values(), axis=0)

    def compute_metrics(self, *args, **kwargs):
        metrics = {}

        for name, approximator in self.approximators.items():
            m = approximator.compute_metrics(*args, **kwargs)
            m = {f"{name}/{key}": value for key, value in m.items()}
            metrics |= m

        return metrics
