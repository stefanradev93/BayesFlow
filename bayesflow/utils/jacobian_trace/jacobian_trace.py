from collections.abc import Callable
import keras

from bayesflow.types import Tensor

from .compute_jacobian_trace import compute_jacobian_trace
from .estimate_jacobian_trace import estimate_jacobian_trace


def jacobian_trace(f: Callable[[Tensor], Tensor], x: Tensor, max_steps: int = 1) -> (Tensor, Tensor):
    """Compute or estimate the trace of the Jacobian matrix of f.

    :param f: The function to be differentiated.

    :param x: Tensor of shape (n, ..., d)
        The input tensor to f.

    :param max_steps: The maximum number of steps to use for the estimate.
        If this does not exceed the dimensionality of f(x), use Hutchinson's algorithm to
        return an unbiased estimate of the Jacobian trace.
        Otherwise, perform an exact computation.
        Default: 1

    :return: 2-tuple of tensors:
        1. The output of f(x)
        2. Tensor of shape (n,)
            An unbiased estimate or the exact trace of the Jacobian of f.
    """
    dims = keras.ops.shape(x)[-1]

    if max_steps is None or dims <= max_steps:
        # use the exact version
        fx, trace = compute_jacobian_trace(f, x)
    else:
        # use an estimate with the maximum number of steps
        fx, trace = estimate_jacobian_trace(f, x, max_steps)

    return fx, trace
