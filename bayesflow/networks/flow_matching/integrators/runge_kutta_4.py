import keras
from bayesflow.types import Tensor, Shape
from bayesflow.utils import find_network, jacobian_trace, keras_kwargs, expand_right_as, tile_axis
from .integrator import Integrator


class RK4Integrator(Integrator):
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
            k1 = self.velocity(network, projector, arg, t, conditions, **kwargs)
            k2 = self.velocity(network, projector, arg + (dt / 2.0 * k1), t + (dt / 2.0), conditions, **kwargs)
            k3 = self.velocity(network, projector, arg + (dt / 2.0 * k2), t + (dt / 2.0), conditions, **kwargs)
            k4 = self.velocity(network, projector, arg + (dt * k3), t + dt, conditions, **kwargs)
            return (k1 + (2 * k2) + (2 * k3) + k4) / 6.0

        if density:
            trace = keras.ops.zeros(keras.ops.shape(x)[:-1], dtype=x.dtype)
            for _ in range(steps):
                v4, tr = jacobian_trace(f, z, kwargs.get("trace_steps", 5))
                z += dt * v4
                trace += dt * tr
                t += dt
            return z, trace

        for _ in range(steps):
            k1 = self.velocity(network, projector, z, t, conditions, **kwargs)
            k2 = self.velocity(network, projector, z + (dt / 2.0 * k1), t + (dt / 2.0), conditions, **kwargs)
            k3 = self.velocity(network, projector, z + (dt / 2.0 * k2), t + (dt / 2.0), conditions, **kwargs)
            k4 = self.velocity(network, projector, z + (dt * k3), t + dt, conditions, **kwargs)
            z += (dt / 6.0) * (k1 + (2 * k2) + (2 * k3) + k4)
            t += dt

        return z
