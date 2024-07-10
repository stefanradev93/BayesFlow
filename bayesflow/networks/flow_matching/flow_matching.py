import keras
from keras.saving import (
    register_keras_serializable,
)

from bayesflow.types import Tensor
from bayesflow.utils import expand_right_as, find_network, jacobian_trace, keras_kwargs, optimal_transport

from ..inference_network import InferenceNetwork


@register_keras_serializable(package="bayesflow.networks")
class FlowMatching(InferenceNetwork):
    def __init__(self, subnet: str = "mlp", base_distribution: str = "normal", **kwargs):
        super().__init__(base_distribution=base_distribution, **keras_kwargs(kwargs))
        self.subnet = find_network(subnet, **kwargs.get("subnet_kwargs", {}))
        self.output_projector = keras.layers.Dense(units=None, bias_initializer="zeros", kernel_initializer="zeros")

        self.seed_generator = keras.random.SeedGenerator()

    def build(self, xz_shape, conditions_shape=None):
        super().build(xz_shape)

        self.output_projector.units = xz_shape[-1]

        xz = keras.ops.zeros(xz_shape)
        if conditions_shape is None:
            conditions = None
        else:
            conditions = keras.ops.zeros(conditions_shape)

        self.call(xz, conditions=conditions, steps=1)

    def call(
        self,
        xz: Tensor,
        conditions: Tensor = None,
        inverse: bool = False,
        **kwargs,
    ):
        if inverse:
            return self._inverse(xz, conditions=conditions, **kwargs)
        return self._forward(xz, conditions=conditions, **kwargs)

    def velocity(self, x: Tensor, t: int | float | Tensor, conditions: Tensor = None) -> Tensor:
        if not keras.ops.is_tensor(t):
            t = keras.ops.convert_to_tensor(t, dtype=x.dtype)

        if keras.ops.ndim(t) == 0:
            t = keras.ops.full((keras.ops.shape(x)[0],), t, dtype=keras.ops.dtype(x))

        t = expand_right_as(t, x)

        if conditions is None:
            xtc = keras.ops.concatenate([x, t], axis=-1)
        else:
            xtc = keras.ops.concatenate([x, t, conditions], axis=-1)

        return self.output_projector(self.subnet(xtc))

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        steps = kwargs.get("steps", 100)
        z = keras.ops.copy(x)
        t = 1.0
        dt = -1.0 / steps

        if density:
            trace = keras.ops.zeros(keras.ops.shape(x)[0], dtype=x.dtype)

            def f(arg):
                return self.velocity(arg, t, conditions)

            for _ in range(steps):
                v, tr = jacobian_trace(f, z, kwargs.get("trace_steps", 5))
                z += dt * v
                trace += dt * tr
                t += dt

            log_prob = self.base_distribution.log_prob(z)

            log_density = log_prob + trace

            return z, log_density
        else:
            for _ in range(steps):
                v = self.velocity(z, t, conditions)
                z += dt * v
                t += dt

            return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        steps = kwargs.get("steps", 100)
        x = keras.ops.copy(z)
        t = 0.0
        dt = 1.0 / steps

        if density:
            trace = keras.ops.zeros(keras.ops.shape(x)[0], dtype=x.dtype)

            def f(arg):
                return self.velocity(arg, t, conditions)

            for _ in range(steps):
                v, tr = jacobian_trace(f, x, kwargs.get("trace_steps", 5))
                x += dt * v
                trace += dt * tr
                t += dt

            log_prob = self.base_distribution.log_prob(z)

            log_density = log_prob - trace

            return x, log_density
        else:
            for _ in range(steps):
                v = self.velocity(x, t, conditions)
                x += dt * v
                t += dt

            return x

    def compute_metrics(self, data: dict[str, Tensor], stage: str = "training") -> dict[str, Tensor]:
        base_metrics = super().compute_metrics(data, stage=stage)

        x1 = data["inference_variables"]
        c = data.get("inference_conditions")

        if not self.built:
            # TODO: the base distribution is not yet built, but we need to sample from it (see below)
            #  ideally, we want to build automatically before this method is called
            xz_shape = keras.ops.shape(x1)
            conditions_shape = None if c is None else keras.ops.shape(c)
            self.build(xz_shape, conditions_shape)

        x0 = self.base_distribution.sample((keras.ops.shape(x1)[0],))

        # TODO: should move this to worker-process somehow
        x0, x1 = optimal_transport(x0, x1, max_steps=int(1e4), regularization=0.01, seed=self.seed_generator)

        t = keras.random.uniform((keras.ops.shape(x0)[0],), seed=self.seed_generator)
        t = expand_right_as(t, x0)

        x = t * x1 + (1 - t) * x0

        predicted_velocity = self.velocity(x, t, c)
        target_velocity = x1 - x0

        loss = keras.losses.mean_squared_error(target_velocity, predicted_velocity)
        loss = keras.ops.mean(loss)

        return base_metrics | {"loss": loss}
