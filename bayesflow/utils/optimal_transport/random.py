import keras
import numpy as np

from bayesflow.types import Tensor


def random(
    x1: Tensor, x2: Tensor, *aux, replace: bool = False, numpy: bool = False, seed: int = None
) -> (Tensor, Tensor):
    if numpy:
        n = x1.shape[0]
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, size=n, replace=replace)

        x1 = np.take(x1, indices, axis=0)
        aux = [np.take(x, indices, axis=0) for x in aux]

        return x1, x2, *aux

    n = keras.ops.shape(x1)[0]

    if replace:
        indices = keras.random.randint((n,), 0, n, seed=seed)
    else:
        indices = keras.random.shuffle(keras.ops.arange(n), seed=seed)

    x1 = keras.ops.take(x1, indices, axis=0)
    aux = [keras.ops.take(x, indices, axis=0) for x in aux]

    return x1, x2, *aux
