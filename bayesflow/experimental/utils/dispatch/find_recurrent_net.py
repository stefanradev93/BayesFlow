
import keras

from functools import singledispatch


@singledispatch
def find_recurrent_net(arg, **kwargs):
    raise TypeError(f"Cannot infer network from {arg!r}.")


@find_recurrent_net.register
def _(name: str, **kwargs):
    match name.lower():
        case "lstm":
            constructor = keras.layers.LSTM
        case "gru":
            constructor = keras.layers.GRU
        case other:
            raise ValueError(f"Unsupported network name: '{other}'.")

    return constructor


@find_recurrent_net.register
def _(network: keras.Layer):
    return network
