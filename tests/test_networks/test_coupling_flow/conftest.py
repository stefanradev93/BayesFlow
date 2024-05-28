
import keras
import pytest


@pytest.fixture()
def actnorm():
    from bayesflow.experimental.networks.coupling_flow.actnorm import ActNorm
    return ActNorm()


@pytest.fixture(params=[2, 3])
def batch_size(request):
    return request.param


@pytest.fixture()
def dual_coupling():
    from bayesflow.experimental.networks.coupling_flow.couplings import DualCoupling
    return DualCoupling.new()


@pytest.fixture(params=["actnorm", "dual_coupling"])
def invertible_layer(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[2, 3])
def num_features(request):
    return request.param


@pytest.fixture()
def random_input(batch_size, num_features):
    return keras.random.normal((batch_size, num_features))


@pytest.fixture()
def single_coupling():
    from bayesflow.experimental.networks.coupling_flow.couplings import SingleCoupling
    return SingleCoupling.new()
