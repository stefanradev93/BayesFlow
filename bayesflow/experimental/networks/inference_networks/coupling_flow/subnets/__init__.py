
import keras

from bayesflow.experimental.networks import ConditionalResidualBlock


def find_subnet(subnet: str | keras.Model | keras.layers.Layer, transform: str, target_dim: int, **kwargs):

    if subnet == "default":
        constructor = ConditionalResidualBlock
    else:
        constructor = subnet

    if transform == "affine":
        output_dim = target_dim * 2
    else:
        raise NotImplementedError

    return constructor(output_dim, **kwargs)
