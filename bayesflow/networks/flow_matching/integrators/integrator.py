import keras
from bayesflow.types import Tensor
from bayesflow.utils import keras_kwargs, expand_right_as, tile_axis


class Integrator:
    def __init__(self, **kwargs):
        super().__init__(**keras_kwargs(kwargs))

    def __call__(self, x: Tensor, steps: int, conditions: Tensor = None, dynamic: bool = False):
        raise NotImplementedError
    
    def velocity(self, network, projector, x: Tensor, t: int | float | Tensor, conditions: Tensor = None, **kwargs):
        if not keras.ops.is_tensor(t):
            t = keras.ops.convert_to_tensor(t, dtype=x.dtype)
        if keras.ops.ndim(t) == 0:
            t = keras.ops.full((keras.ops.shape(x)[0],), t, dtype=keras.ops.dtype(x))

        t = expand_right_as(t, x)
        if keras.ops.ndim(x) == 3:
            t = tile_axis(t, n=keras.ops.shape(x)[1], axis=1)

        if conditions is None:
            xtc = keras.ops.concatenate([x, t], axis=-1)
        else:
            xtc = keras.ops.concatenate([x, t, conditions], axis=-1)

        return projector(network(xtc, **kwargs))
