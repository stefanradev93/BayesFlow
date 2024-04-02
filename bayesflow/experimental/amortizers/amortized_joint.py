
import keras

from .amortizer import Amortizer


class JointAmortizer(keras.Model):
    def __init__(self, **amortizers: Amortizer):
        super().__init__()
        self.amortizers = amortizers

    def call(self, *args, **kwargs):
        return {name: amortizer(*args, **kwargs) for name, amortizer in self.amortizers.items()}

    def compute_loss(self, *args, **kwargs):
        losses = {name: amortizer.compute_loss(*args, **kwargs) for name, amortizer in self.amortizers.items()}
        return keras.ops.mean(losses.values(), axis=0)

    def compute_metrics(self, *args, **kwargs):
        metrics = {}

        for name, amortizer in self.amortizers.items():
            m = amortizer.compute_metrics(*args, **kwargs)
            m = {f"{name}/{key}": value for key, value in m.items()}
            metrics |= m

        return metrics
