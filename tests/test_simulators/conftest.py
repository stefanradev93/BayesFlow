import keras
import numpy as np
import pytest

from bayesflow.simulators import SequentialSimulator


@pytest.fixture()
def batch_size():
    return 16


@pytest.fixture(params=[False, True], autouse=True)
def use_batched(request):
    return request.param


@pytest.fixture(params=[False, True], autouse=True)
def use_numpy(request):
    return request.param


@pytest.fixture(params=[False, True], autouse=True)
def use_squeezed(request):
    return request.param


@pytest.fixture()
def sequential_two_moons_numpy():
    def contexts():
        r = np.random.normal(0.1, 0.01)
        alpha = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)
        return dict(r=r, alpha=alpha)

    def parameters():
        return dict(theta=np.random.uniform(-1.0, 1.0, size=2))

    def observables(r, alpha, theta):
        print(f"{r.shape=}, {alpha.shape=}, {theta[0].shape=}")
        x1 = -np.abs(theta[0] + theta[1]) / np.sqrt(2.0) + r * np.cos(alpha) + 0.25
        x2 = (-theta[0] + theta[1]) / np.sqrt(2.0) + r * np.sin(alpha)
        return dict(x=np.stack([x1, x2]))

    return SequentialSimulator([contexts, parameters, observables])


@pytest.fixture()
def sequential_two_moons_keras():
    def contexts(batch_shape):
        r = keras.random.normal(batch_shape, 0.1, 0.01)
        alpha = keras.random.uniform(batch_shape, -0.5 * np.pi, 0.5 * np.pi)

        return dict(r=r, alpha=alpha)

    def parameters(batch_shape):
        theta = keras.random.normal(batch_shape + (2,), -1.0, 1.0)
        return dict(theta=theta)

    def observables(batch_shape, r, alpha, theta):
        print(f"{r.shape=}, {alpha.shape=}, {theta[:, 0].shape=}")
        x1 = -keras.ops.abs(theta[:, 0] + theta[:, 1]) / np.sqrt(2.0) + r * keras.ops.cos(alpha) + 0.25
        x2 = (-theta[:, 0] + theta[:, 1]) / np.sqrt(2.0) + r * keras.ops.sin(alpha)
        return dict(x=keras.ops.stack([x1, x2], axis=1))

    return SequentialSimulator([contexts, parameters, observables])


@pytest.fixture(params=["sequential_two_moons_keras", "sequential_two_moons_numpy", "two_moons"])
def simulator(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def two_moons():
    from bayesflow.simulators import TwoMoonsSimulator

    return TwoMoonsSimulator()
