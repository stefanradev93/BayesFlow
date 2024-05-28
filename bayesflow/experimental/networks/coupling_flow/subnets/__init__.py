
# TODO: This is a temporary solution to avoid circular imports.
#  The correct solution is to move the subnets to a separate module.
#  We should also probably use "ResNet" here
from typing import Callable

from ..transforms import Transform
from ...resnet.residual_block import ConditionalResidualBlock


def find_subnet(subnet: str | Callable, transform: Transform, output_dim: int, **kwargs):

    match subnet:
        case str() as name:
            match name.lower():
                case "default":
                    constructor = ConditionalResidualBlock
                case other:
                    raise NotImplementedError(f"Unsupported subnet name: '{other}'.")
        case callable():
            constructor = subnet
        case other:
            raise NotImplementedError(f"Cannot infer subnet constructor from {other!r}.")
    output_dim = output_dim * transform.params_per_dim
    return constructor(output_dim, **kwargs)
