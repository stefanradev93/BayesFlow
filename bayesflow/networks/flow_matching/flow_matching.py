import keras
from keras.saving import (
    register_keras_serializable,
)

from bayesflow.types import Tensor
from bayesflow.utils import expand_right_as, find_network, jacobian_trace, keras_kwargs, optimal_transport, tile_axis

from ..inference_network import InferenceNetwork


@register_keras_serializable(package="bayesflow.networks")
class FlowMatching(InferenceNetwork):
    """Implements Optimal Transport Flow Matching, originally introduced as Rectified Flow,
    with ideas incorporated from [1-3].

    [1] Rectified Flow: arXiv:2209.03003
    [2] Flow Matching: arXiv:2210.02747
    [3] Optimal Transport Flow Matching: arXiv:2302.00482
    """

    def __init__(self, subnet: str = "mlp", base_distribution: str = "normal", **kwargs):
        super().__init__(base_distribution=base_distribution, **keras_kwargs(kwargs))
        self.subnet = find_network(subnet, **kwargs.get("subnet_kwargs", {}))
        self.output_projector = keras.layers.Dense(units=None, bias_initializer="zeros", kernel_initializer="zeros")

        self.seed_generator = keras.random.SeedGenerator()

    def build(self, xz_shape, conditions_shape=None):
        super().build(xz_shape)

        self.output_projector.units = xz_shape[-1]

        input_shape = list(xz_shape)

        # time vector
        input_shape[-1] += 1

        if conditions_shape is not None:
            input_shape[-1] += conditions_shape[-1]

        input_shape = tuple(input_shape)

        self.subnet.build(input_shape)

        input_shape = self.subnet.compute_output_shape(input_shape)
        self.output_projector.build(input_shape)

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

    def velocity(self, x: Tensor, t: int | float | Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        if not keras.ops.is_tensor(t):
            t = keras.ops.convert_to_tensor(t, dtype=x.dtype)

        if keras.ops.ndim(t) == 0:
            t = keras.ops.full((keras.ops.shape(x)[0],), t, dtype=keras.ops.dtype(x))

        t = expand_right_as(t, x)
        if keras.ops.ndim(x) == 3:
            t = tile_axis(t, axis=1, n=keras.ops.shape(x)[1])

        if conditions is None:
            xtc = keras.ops.concatenate([x, t], axis=-1)
        else:
            xtc = keras.ops.concatenate([x, t, conditions], axis=-1)

        return self.output_projector(self.subnet(xtc, **kwargs))

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
                return self.velocity(arg, t, conditions, **kwargs)

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
                v = self.velocity(z, t, conditions, **kwargs)
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

    def compute_metrics(self, x1: Tensor, conditions: Tensor = None, stage: str = "training") -> dict[str, Tensor]:
        base_metrics = super().compute_metrics(x1, conditions=conditions, stage=stage)

        batch_size = keras.ops.shape(x1)[0]
        x0 = self.base_distribution.sample((batch_size,))

        x0, x1 = optimal_transport(x0, x1, max_steps=int(1e4), regularization=0.01, seed=self.seed_generator)

        t = keras.random.uniform((batch_size, 1), seed=self.seed_generator)
        t = expand_right_as(t, x0)

        x = t * x1 + (1 - t) * x0

        predicted_velocity = self.velocity(x, t, conditions)
        target_velocity = x1 - x0

        loss = keras.losses.mean_squared_error(target_velocity, predicted_velocity)
        loss = keras.ops.mean(loss)

        return base_metrics | {"loss": loss}
