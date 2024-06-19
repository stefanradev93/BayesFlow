
from typing import Tuple, Union

import keras
from keras.saving import (
    register_keras_serializable,
)
from scipy.integrate import solve_ivp

from bayesflow.types import Tensor
from bayesflow.utils import find_network, keras_kwargs

from ..inference_network import InferenceNetwork


@register_keras_serializable(package="bayesflow.networks")
class FlowMatching(InferenceNetwork):
    def __init__(self, network: str = "resnet", **kwargs):
        super().__init__(**keras_kwargs(kwargs))
        self.network = find_network(network, **kwargs)

    def velocity(self, x: Tensor, t: Tensor, conditions: any = None):
        if conditions is None:
            xtc = keras.ops.concatenate([x, t], axis=1)
        else:
            xtc = keras.ops.concatenate([x, t, conditions], axis=1)

        return self.network(xtc)

    def _forward(self, x: Tensor, conditions: any = None, jacobian: bool = False, steps: int = 100, method: str = "RK45") -> Union[Tensor, Tuple[Tensor, Tensor]]:
        def dfdt(t: float, x: Tensor):
            t = keras.ops.full((keras.ops.shape(x)[0], 1), t)
            return self.velocity(x, t, conditions)

        return solve_ivp(dfdt, t_span=(1.0, 0.0), y0=x, method=method, vectorized=True)[1]

    def _inverse(self, z: Tensor, conditions: any = None, jacobian: bool = False, steps: int = 100, method: str = "RK45") -> Union[Tensor, Tuple[Tensor, Tensor]]:
        def dfdt(t: float, x: Tensor):
            t = keras.ops.full((keras.ops.shape(x)[0], 1), t)
            return self.velocity(x, t, conditions)

        return solve_ivp(dfdt, t_span=(0.0, 1.0), y0=z, method=method, vectorized=True)[1]

    def compute_loss(self, x=None, **kwargs):
        x0, x1, *conditions = x
        t = keras.random.uniform((keras.ops.shape(x0)[0], 1))

        x = t * x1 + (1 - t) * x0
        xtc = keras.ops.concatenate([x, t, *conditions], axis=-1)

        predicted_velocity = self.network(xtc)
        target_velocity = x1 - x0

        return keras.losses.mean_squared_error(predicted_velocity, target_velocity)


# @register_keras_serializable(package="bayesflow.networks")
# class FlowMatching(InferenceNetwork):
#     def __init__(self, network: keras.Layer, **kwargs):
#         super().__init__(**kwargs)
#         self.network = network
#
#     @classmethod
#     def new(cls, network: str = "resnet", base_distribution: str = "normal"):
#         # TODO: we probably want to provide a factory method like this, since the other networks use it
#         #  for high-level input parameters
#         # network = find_network(network)
#         return cls(network, base_distribution=base_distribution)
#
#     @classmethod
#     def from_config(cls, config: dict, custom_objects=None) -> "FlowMatching":
#         # TODO: the base distribution must be savable and loadable
#         #  ideally we also don't want to have to manually deserialize it in every subclass of InferenceNetwork
#         base_distribution = deserialize_keras_object(config.pop("base_distribution"))
#         network = deserialize_keras_object(config.pop("network"))
#         return cls(network, base_distribution=base_distribution, **config)
#
#     def get_config(self) -> dict:
#         base_config = super().get_config()
#         config = {"network": serialize_keras_object(self.network)}
#         return base_config | config
#
#     def build(self, input_shape):
#         self.network.build(input_shape)
#
#     def _forward(self, x: Tensor, conditions: any = None, jacobian: bool = False, steps: int = 100, method: str = "RK45") -> Union[Tensor, Tuple[Tensor, Tensor]]:
#         # implement conditions = None and jacobian = False first
#         # then work your way up
#         raise NotImplementedError
#
#     def _inverse(self, z: Tensor, conditions: any = None, jacobian: bool = False, steps: int = 100, method: str = "RK45") -> Union[Tensor, Tuple[Tensor, Tensor]]:
#         raise NotImplementedError
#
#     def compute_loss(self, x=None, **kwargs):
#         # x should ideally contain both x0 and x1,
#         # where the optimal transport matching already happened in the worker process
#         # this is possible, but might not be super user-friendly. We will have to see.
#         x0, x1, t = x
#
#         xt = t * x1 + (1 - t) * x0
#
#         # get velocity at xt
#         v = ...
#
#         # target velocity:
#         vstar = x1 - x0
#
#         # return mse between v and vstar
#
#
# # TODO: see below for reference implementation
#
#
# class FlowMatching(keras.Model):
#     def __init__(self, network: keras.Layer, base_distribution):
#         super().__init__()
#         self.network = network
#         self.base_distribution = find_distribution(base_distribution)
#
#     def call(self, inferred_variables, inference_conditions):
#         return self.network(keras.ops.concatenate([inferred_variables, inference_conditions], axis=1))
#
#     def compute_loss(self, x=None, y=None, y_pred=None, **kwargs):
#         return keras.losses.mean_squared_error(y, y_pred)
#
#     def velocity(self, x: Tensor, t: Tensor, c: Tensor = None):
#         if c is None:
#             xtc = keras.ops.concatenate([x, t], axis=1)
#         else:
#             xtc = keras.ops.concatenate([x, t, c], axis=1)
#
#         return self.network(xtc)
#
#     def forward(self, x, c=None, method="RK45") -> Tensor:
#         def f(t, x):
#             t = keras.ops.full((keras.ops.shape(x)[0], 1), t)
#             return self.velocity(x, t, c)
#
#         bunch = solve_ivp(f, t_span=(1.0, 0.0), y0=x, method=method, vectorized=True)
#
#         return bunch[1]
#
#     def inverse(self, x, c=None, method="RK45") -> Tensor:
#         def f(t, x):
#             t = keras.ops.full((keras.ops.shape(x)[0], 1), t)
#             return self.velocity(x, t, c)
#
#         bunch = solve_ivp(f, t_span=(0.0, 1.0), y0=x, method=method, vectorized=True)
#
#         return bunch[1]
#
#     def sample(self, batch_shape: Shape) -> Tensor:
#         z = self.base_distribution.sample(batch_shape)
#         return self.inverse(z)
#
#     def log_prob(self, x: Tensor, c: Tensor = None) -> Tensor:
#         raise NotImplementedError(f"Keras does not yet support backend-agnostic Vector-Jacobian Products.")
#
#
# def hutchinson_trace(f: callable, x: Tensor) -> (Tensor, Tensor):
#     # TODO: test this for all 3 backends
#     noise = keras.random.normal(keras.ops.shape(x))
#
#     match keras.backend.backend():
#         case "jax":
#             import jax
#             fx, jvp = jax.jvp(f, (x,), (noise,))
#         case "tensorflow":
#             import tensorflow as tf
#             with tf.GradientTape(persistent=True) as tape:
#                 tape.watch(x)
#                 fx = f(x)
#             jvp = tape.gradient(fx, x, output_gradients=noise)
#         case "torch":
#             import torch
#             fx, jvp = torch.autograd.functional.jvp(f, x, noise, create_graph=True)
#         case other:
#             raise NotImplementedError(f"Backend {other} is not supported for trace estimation.")
#
#     trace = keras.ops.sum(jvp * noise, axis=1)
#
#     return fx, trace
