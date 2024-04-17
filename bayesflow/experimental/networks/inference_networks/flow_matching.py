
import keras
from scipy.integrate import solve_ivp

from bayesflow.experimental.types import Tensor


class FlowMatching(keras.Model):
    def __init__(self, network: keras.Layer, base_distribution):
        super().__init__()
        self.network = network
        self.base_distribution = find_distribution(base_distribution)

    def call(self, inferred_variables, inference_conditions):
        return self.network(keras.ops.concatenate([inferred_variables, inference_conditions], axis=1))

    def compute_loss(self, x=None, y=None, y_pred=None, **kwargs):
        return keras.losses.mean_squared_error(y, y_pred)

    def velocity(self, x: Tensor, t: Tensor, c: Tensor = None):
        if c is None:
            xtc = keras.ops.concatenate([x, t], axis=1)
        else:
            xtc = keras.ops.concatenate([x, t, c], axis=1)

        return self.network(xtc)

    def forward(self, x, c=None, method="RK45") -> Tensor:
        def f(t, x):
            t = keras.ops.full((keras.ops.shape(x)[0], 1), t)
            return self.velocity(x, t, c)

        bunch = solve_ivp(f, t_span=(1.0, 0.0), y0=x, method=method, vectorized=True)

        return bunch[1]

    def inverse(self, x, c=None, method="RK45") -> Tensor:
        def f(t, x):
            t = keras.ops.full((keras.ops.shape(x)[0], 1), t)
            return self.velocity(x, t, c)

        bunch = solve_ivp(f, t_span=(0.0, 1.0), y0=x, method=method, vectorized=True)

        return bunch[1]

    def sample(self, batch_shape: Shape) -> Tensor:
        z = self.base_distribution.sample(batch_shape)
        return self.inverse(z)

    def log_prob(self, x: Tensor, c: Tensor = None) -> Tensor:
        raise NotImplementedError(f"Keras does not yet support backend-agnostic Vector-Jacobian Products.")


def hutchinson_trace(f: callable, x: Tensor) -> (Tensor, Tensor):
    # TODO: test this for all 3 backends
    noise = keras.random.normal(keras.ops.shape(x))

    match keras.backend.backend():
        case "jax":
            import jax
            fx, jvp = jax.jvp(f, (x,), (noise,))
        case "tensorflow":
            import tensorflow as tf
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                fx = f(x)
            jvp = tape.gradient(fx, x, output_gradients=noise)
        case "torch":
            import torch
            fx, jvp = torch.autograd.functional.jvp(f, x, noise, create_graph=True)
        case other:
            raise NotImplementedError(f"Backend {other} is not supported for trace estimation.")

    trace = keras.ops.sum(jvp * noise, axis=1)

    return fx, trace
