
from bayesflow.experimental.backend_agnostic import SampleObservationsMixin


class Simulator(SampleObservationsMixin):
    """
    Migration Decorator for SampleObservationsMixin

    Example Usage:
    ```py3
    @Simulator
    def simulator(batch_shape, /, *, parameters, contexts=None):
        x1 = parameters["param1"] + contexts["context1"]
        x2 = parameters["param2"] - contexts["context2"]

        return {"x1": x1, "x2": x2}
    ```
    """
    def __init__(self, sample_fn: callable):
        self.sample_fn = sample_fn

    def __call__(self, *args, **kwargs):
        return self.sample_fn(*args, **kwargs)

    def sample_observations(self, batch_shape, /, *, parameters, contexts=None):
        return self.sample_fn(batch_shape, parameters, contexts)
