from collections.abc import Callable
import keras
import numpy as np

from bayesflow.types import Tensor


from ._vjp import _make_vjp_fn


def compute_jacobian_trace(f: Callable[[Tensor], Tensor], x: Tensor) -> (Tensor, Tensor):
    """Compute the exact trace of the Jacobian matrix of f by projection on each axis.

    :param f: The function to be differentiated.

    :param x: Tensor of shape (n, ..., d)
        The input tensor to f.

    :return: 2-tuple of tensors:
        1. The output of f(x)
        2. Tensor of shape (n,)
            The exact trace of the Jacobian matrix of f.
    """
    shape = keras.ops.shape(x)
    trace = keras.ops.zeros(shape[:-1])

    fx, vjp_fn = _make_vjp_fn(f, x)

    for dim in range(shape[-1]):
        projector = np.zeros(shape, dtype="float32")
        projector[..., dim] = 1.0
        projector = keras.ops.convert_to_tensor(projector)

        vjp = vjp_fn(projector)

        trace += vjp[..., dim]

    return fx, trace
