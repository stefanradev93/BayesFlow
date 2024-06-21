import keras
import numpy as np

from bayesflow.types import Tensor


def jacobian_trace(f: callable, x: Tensor, samples: int = 1) -> (Tensor, Tensor):
    """
    Returns an unbiased estimate of the trace of the Jacobian of f, using Hutchinson's estimator.

    :param f: The function to be differentiated.
        Must take x as its only argument and return a single output Tensor.

    :param x: Tensor of shape (n, d)
        The input tensor to f.

    :param samples: The number of random samples to use for the estimate.
        If this exceeds the dimensionality of f(x) or you pass None, we
        will instead perform an exact computation which takes that many samples.
        Default: 1

    :return: Tensor of shape (n,)
        An unbiased estimate of the trace of the Jacobian of f.
    """
    # copy here to avoid causing outside side effects
    # TODO: this may not be necessary for every backend
    x = keras.ops.copy(x)
    batch_size, dims = keras.ops.shape(x)

    match keras.backend.backend():
        case "jax":
            import jax

            fx, vjp_fn = jax.vjp(f, x)
            vjp_fn = jax.jit(vjp_fn)

            trace = keras.ops.zeros((batch_size,), dtype=x.dtype)

            # TODO: can we use jax.vmap to avoid the for loop?

            if samples is None or dims <= samples:
                # exact
                for dim in range(dims):
                    projector = keras.ops.zeros((batch_size, dims), dtype=x.dtype)
                    projector = projector.at[:, dim].set(1.0)

                    vjp = vjp_fn(projector)[0]

                    trace += vjp[:, dim]
            else:
                # estimate
                for sample in range(samples):
                    projector = keras.random.normal((batch_size, dims), dtype=x.dtype)

                    vjp = vjp_fn(projector)[0]

                    trace += keras.ops.sum(vjp * projector, axis=1)

        case "tensorflow":
            import tensorflow as tf

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                fx = f(x)

            trace = keras.ops.zeros((batch_size,))

            # TODO: can we use tf.gradients to avoid the for loop?

            if samples is None or dims <= samples:
                # exact
                for dim in range(dims):
                    projector = np.zeros((batch_size, dims), dtype=keras.backend.standardize_dtype(x.dtype))
                    projector[:, dim] = 1.0
                    projector = keras.ops.convert_to_tensor(projector)

                    vjp = tape.gradient(fx, x, projector)

                    trace += vjp[:, dim]
            else:
                # estimate
                for _ in range(samples):
                    projector = keras.random.normal((batch_size, dims), dtype=x.dtype)

                    vjp = tape.gradient(fx, x, projector)

                    trace += keras.ops.sum(vjp * projector, axis=1) / samples
        case "torch":
            import torch

            x.requires_grad_(True)

            with torch.enable_grad():
                fx = f(x)

            trace = keras.ops.zeros(keras.ops.shape(x)[0])

            # TODO: can we use is_grads_batched to avoid the for loop?

            if samples is None or dims <= samples:
                # exact
                for dim in range(dims):
                    projector = keras.ops.zeros((batch_size, dims), dtype=x.dtype)
                    projector[:, dim] = 1.0

                    vjp = torch.autograd.grad(fx, x, projector, retain_graph=True)[0]

                    trace += vjp[:, dim]
            else:
                # estimate
                for _ in range(samples):
                    projector = keras.random.normal((batch_size, dims), dtype=x.dtype)

                    vjp = torch.autograd.grad(fx, x, projector, retain_graph=True)[0]

                    trace += keras.ops.sum(vjp * projector, axis=1) / samples
        case other:
            raise NotImplementedError(f"Jacobian trace computation is currently not supported for backend '{other}'.")

    return fx, trace
