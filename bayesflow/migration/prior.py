
from bayesflow.experimental.backend_agnostic import SampleParametersMixin


class Prior(SampleParametersMixin):
    """
    Migration Decorator for SampleParametersMixin

    Example Usage:
    ```py3
    @Prior
    def prior(batch_shape, /, *, contexts=None):
        param1 = tf.random.normal(batch_shape, 0, 1)
        param2 = tf.random.uniform(batch_shape, -1, 1)

        return {"param1": param1, "param2": param2}
    ```
    """
    def __init__(self, sample_fn: callable):
        self.sample_fn = sample_fn

    def __call__(self, *args, **kwargs):
        return self.sample_fn(*args, **kwargs)

    def sample_parameters(self, batch_shape, /, *, contexts=None):
        return self.sample_fn(batch_shape, contexts)
