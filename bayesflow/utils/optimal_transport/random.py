import keras
import numpy as np

from bayesflow.types import Tensor


def random(x1: Tensor, x2: Tensor, replace: bool = False, numpy: bool = False, seed: int = None) -> (Tensor, Tensor):
    if numpy:
        if seed is not None:
            np.random.seed(seed)

        indices = np.random.choice(len(x2), size=len(x2), replace=replace)

        return x1, x2[indices]

    if replace:
        indices = keras.random.randint((len(x2),), 0, len(x2), seed=seed)
        x2 = keras.ops.take(x2, indices, axis=0)
    else:
        x2 = keras.random.shuffle(x2, axis=0, seed=seed)

    return x1, x2
