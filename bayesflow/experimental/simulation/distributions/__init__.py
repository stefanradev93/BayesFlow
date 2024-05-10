
import keras
from bayesflow.experimental.types import Distribution, Shape

from .joint_distribution import JointDistribution


def find_distribution(distribution: str | Distribution | type(Distribution), shape: Shape) -> Distribution:
    if isinstance(distribution, Distribution):
        return distribution
    if isinstance(distribution, type):
        return Distribution()

    match distribution:
        case "normal":
            match keras.backend.backend():
                case "jax" | "tensorflow":
                    import tensorflow as tf
                    import tensorflow_probability as tfp
                    distribution = tfp.distributions.Normal(tf.zeros(shape), tf.ones(shape))
                    distribution = tfp.distributions.Independent(distribution, 1)
                case "torch":
                    import torch
                    import torch.distributions as D
                    distribution = D.Normal(torch.zeros(shape), torch.ones(shape))
                    distribution = D.Independent(distribution, 1)
        case str() as unknown_distribution:
            raise ValueError(f"Distribution '{unknown_distribution}' is unknown or not yet supported by name.")
        case other:
            raise TypeError(f"Unknown distribution type: {other}")

    return distribution
