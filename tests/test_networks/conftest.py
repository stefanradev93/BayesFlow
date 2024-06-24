import keras
import pytest


@pytest.fixture(params=[2, 3])
def batch_size(request):
    return request.param


@pytest.fixture()
def coupling_flow():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(depth=2, subnet_kwargs=dict(depth=2, width=64))


@pytest.fixture()
def flow_matching():
    from bayesflow.networks import FlowMatching

    return FlowMatching(network_kwargs=dict(depth=2, width=64))


@pytest.fixture(params=["coupling_flow"])
def inference_network(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["inference_network", "summary_network"])
def network(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[2, 3])
def num_conditions(request):
    return request.param


@pytest.fixture(params=[2, 3])
def num_features(request):
    return request.param


@pytest.fixture(params=[False, True])
def random_conditions(request, batch_size, num_conditions):
    if not request.param:
        return None

    return keras.random.normal((batch_size, num_conditions))


@pytest.fixture()
def random_samples(batch_size, num_features):
    return keras.random.normal((batch_size, num_features))


@pytest.fixture()
def random_set(batch_size, set_size, num_features):
    return keras.random.normal((batch_size, set_size, num_features))


@pytest.fixture()
def resnet():
    from bayesflow.networks import ResNet

    return ResNet()


@pytest.fixture(params=[2, 3])
def set_size(request):
    return request.param


@pytest.fixture(params=[])
def summary_network(request):
    return request.getfixturevalue(request.param)
