
from bayesflow.experimental.backend_agnostic import SampleContextsMixin


class ContextGenerator(SampleContextsMixin):
    """
    Migration Decorator for SampleContextsMixin

    Example Usage:
    ```py3
    @ContextGenerator
    def context_generator(batch_shape, /):
        context1 = tf.random.normal(batch_shape, 0, 1)
        context2 = tf.random.uniform(batch_shape, -1, 1)

        return {"context1": context1, "context2": context2}
    ```
    """
    def __init__(self, sample_fn: callable):
        self.sample_fn = sample_fn

    def __call__(self, *args, **kwargs):
        return self.sample_fn(*args, **kwargs)

    def sample_contexts(self, batch_shape, /):
        return self.sample_fn(batch_shape)
