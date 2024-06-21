import keras
import numpy as np
import pytest


@pytest.fixture()
def batch_size():
    return 16


@pytest.fixture()
def sequential_two_moons():
    from bayesflow.simulators import SequentialSimulator

    def context():
        r = keras.random.normal((), 0.1, 0.01)
        alpha = keras.random.uniform((), -0.5 * np.pi, 0.5 * np.pi)
        return dict(r=r, alpha=alpha)

    def parameters():
        theta = keras.random.uniform((2,), -1.0, 1.0)
        return dict(theta=theta)

    def observables(r, alpha, theta):
        x1 = -keras.ops.abs(theta[..., :1] + theta[..., 1:]) / np.sqrt(2.0) + r * keras.ops.cos(alpha) + 0.25
        x2 = (-theta[..., :1] + theta[..., 1:]) / np.sqrt(2.0) + r * keras.ops.sin(alpha)

        x = keras.ops.concatenate([x1, x2], axis=-1)

        return dict(x=x)

    return SequentialSimulator([context, parameters, observables])


@pytest.fixture(params=["sequential_two_moons", "two_moons"])
def simulator(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def two_moons():
    from bayesflow.simulators import TwoMoonsSimulator

    return TwoMoonsSimulator()
