import keras

from bayesflow.types import Tensor


def _make_vjp_fn(f: callable, x: Tensor) -> (Tensor, callable):
    match keras.backend.backend():
        case "jax":
            import jax

            fx, _vjp_fn = jax.vjp(f, x)

            @jax.jit
            def vjp_fn(projector):
                return _vjp_fn(projector)[0]
        case "tensorflow":
            import tensorflow as tf

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                fx = f(x)

            def vjp_fn(projector):
                return tape.gradient(fx, x, projector)
        case "torch":
            import torch

            x = keras.ops.copy(x)
            x.requires_grad_(True)

            with torch.enable_grad():
                fx = f(x)

            def vjp_fn(projector):
                return torch.autograd.grad(fx, x, projector, retain_graph=True)[0]
        case other:
            raise NotImplementedError(f"Cannot build a vjp function for backend '{other}'.")

    return fx, vjp_fn
