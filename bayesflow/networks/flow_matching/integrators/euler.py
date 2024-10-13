import keras
from bayesflow.types import Tensor, Shape
from bayesflow.utils import find_network, jacobian_trace, keras_kwargs, expand_right_as, tile_axis
from .integrator import Integrator


class EulerIntegrator(Integrator):
    """
    TODO: docstring
    """

    def __init__(self, **kwargs):
        super().__init__(**keras_kwargs(kwargs))

    def __call__(
        self,
        network,
        projector,
        x: Tensor,
        conditions: Tensor = None,
        steps: int = 100,
        density: bool = False,
        inverse: bool = False,
        **kwargs,
    ):
        z = keras.ops.copy(x)
        t = 1.0 if not inverse else 0.0
        dt = -1.0 / steps if not inverse else 1.0 / steps

        def f(arg):
            return self.velocity(network, projector, arg, t, conditions, **kwargs)

        if density:
            trace = keras.ops.zeros(keras.ops.shape(x)[:-1], dtype=x.dtype)
            for _ in range(steps):
                v, tr = jacobian_trace(f, z, kwargs.get("trace_steps", 5))
                z += dt * v
                trace += dt * tr
                t += dt
            return z, trace

        for _ in range(steps):
            v = self.velocity(network, projector, z, t, conditions, **kwargs)
            z += dt * v
            t += dt

        return z
