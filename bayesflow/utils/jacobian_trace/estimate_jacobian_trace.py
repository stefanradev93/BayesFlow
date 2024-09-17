import keras

from bayesflow.types import Tensor

from ._vjp import _make_vjp_fn


def estimate_jacobian_trace(f: callable, x: Tensor, steps: int = 1) -> (Tensor, Tensor):
    """Estimate the trace of the Jacobian matrix of f using Hutchinson's algorithm.

    :param f: The function to be differentiated.

    :param x: Tensor of shape (n,..., d)
        The input tensor to f.

    :param steps: The number of steps to use for the estimate.
        Higher values yield better precision.
        Default: 1

    :return: 2-tuple of tensors:
        1. The output of f(x)
        2. Tensor of shape (n,)
            An unbiased estimate of the trace of the Jacobian matrix of f.
    """
    shape = keras.ops.shape(x)
    trace = keras.ops.zeros(shape[:-1])

    fx, vjp_fn = _make_vjp_fn(f, x)

    for _ in range(steps):
        projector = keras.random.normal(shape)

        vjp = vjp_fn(projector)

        trace += keras.ops.sum(vjp * projector, axis=-1)

    return fx, trace
