import keras
from keras.saving import (
    register_keras_serializable,
)

from bayesflow.types import Tensor
from bayesflow.utils import find_network, jacobian_trace, keras_kwargs, optimal_transport

from ..inference_network import InferenceNetwork


@register_keras_serializable(package="bayesflow.networks")
class FlowMatching(InferenceNetwork):
    def __init__(self, subnet: str = "resnet", base_distribution: str = "normal", **kwargs):
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
        t = keras.ops.convert_to_tensor(t, dtype=x.dtype)
        match keras.ops.ndim(t):
            case 0:
                t = keras.ops.full((keras.ops.shape(x)[0], 1), t, dtype=x.dtype)
            case 1:
                t = keras.ops.expand_dims(t, 1)

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
        t = keras.ops.ones((keras.ops.shape(x)[0], 1), dtype=x.dtype)
        dt = -1.0 / steps

        if density:
            trace = keras.ops.zeros(keras.ops.shape(x)[0], dtype=x.dtype)

            def f(arg):
                return self.velocity(arg, t, conditions)

            for _ in range(steps):
                v, tr = jacobian_trace(f, z, kwargs.get("trace_steps", 5))
                z += dt * v
                trace += dt * tr

            log_prob = self.base_distribution.log_prob(z)

            log_density = log_prob + trace

            return z, log_density
        else:
            for _ in range(steps):
                v = self.velocity(z, t, conditions)
                z += dt * v

            return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        steps = kwargs.get("steps", 100)
        x = keras.ops.copy(z)
        t = keras.ops.zeros((keras.ops.shape(x)[0], 1), dtype=x.dtype)
        dt = 1.0 / steps

        if density:
            trace = keras.ops.zeros(keras.ops.shape(x)[0], dtype=x.dtype)

            def f(arg):
                return self.velocity(arg, t, conditions)

            for _ in range(steps):
                v, tr = jacobian_trace(f, x, kwargs.get("trace_steps", 5))
                x += dt * v
                trace += dt * tr

            log_prob = self.base_distribution.log_prob(z)

            log_density = log_prob - trace

            return x, log_density
        else:
            for _ in range(steps):
                v = self.velocity(x, t, conditions)
                x += dt * v

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
        x0, x1 = optimal_transport(x0, x1)

        t = keras.random.uniform((keras.ops.shape(x0)[0], 1), seed=self.seed_generator)

        x = t * x1 + (1 - t) * x0

        predicted_velocity = self.velocity(x, t, c)
        target_velocity = x1 - x0

        loss = keras.losses.mean_squared_error(predicted_velocity, target_velocity)
        loss = keras.ops.mean(loss)

        return base_metrics | {"loss": loss}
