
import keras


def isclose(x1, x2, rtol=1e-5, atol=1e-8):
    return keras.ops.abs(x1 - x2) <= atol + rtol * keras.ops.abs(x2)


def allclose(x1, x2, rtol=1e-5, atol=1e-8):
    return keras.ops.all(isclose(x1, x2, rtol, atol))
