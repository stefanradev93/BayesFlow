import keras
from keras.saving import (
    register_keras_serializable,
)
from bayesflow.types import Tensor
from bayesflow.utils import expand_right_as, find_network, jacobian_trace, keras_kwargs, optimal_transport, tile_axis
from ..inference_network import InferenceNetwork
from .integrators import EulerIntegrator
from .integrators import RK2Integrator
from .integrators import RK4Integrator


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
        self.seed_generator = keras.random.SeedGenerator()
        self.integrator = EulerIntegrator(subnet, **kwargs)


    def build(self, xz_shape, conditions_shape=None):
        super().build(xz_shape)
        self.integrator.build(xz_shape, conditions_shape)


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
        
        
    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        steps = kwargs.get("steps", 200)

        if density:
            z, trace = self.integrator(x, conditions=conditions, steps=steps, traced=True)
            log_prob = self.base_distribution.log_prob(z)
            log_density = log_prob + trace
            return z, log_density
        
        z = self.integrator(x, conditions=conditions, steps=steps, traced=False)
        return z


    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        steps = kwargs.get("steps", 100)
        
        if density:
            x, trace = self.integrator(z, conditions=conditions, steps=steps, traced=True, inverse=True)
            log_prob = self.base_distribution.log_prob(z)
            log_density = log_prob - trace
            return x, log_density
        
        x = self.integrator(z, conditions=conditions, steps=steps, traced=False, inverse=True)
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

        predicted_velocity = self.integrator.velocity(x, t, c)
        target_velocity = x1 - x0

        loss = keras.losses.mean_squared_error(target_velocity, predicted_velocity)
        loss = keras.ops.mean(loss)

        return base_metrics | {"loss": loss}
