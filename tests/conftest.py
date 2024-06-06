
import keras
import pytest


@pytest.fixture()
def amortizer(inference_network, summary_network):
    from bayesflow.experimental.amortizers import Amortizer

    return Amortizer(
        inference_network=inference_network,
        summary_network=summary_network,
    )


@pytest.fixture()
def coupling_flow():
    from bayesflow.experimental.networks import CouplingFlow
    return CouplingFlow()


@pytest.fixture()
def flow_matching():
    from bayesflow.experimental.networks import FlowMatching
    return FlowMatching()


@pytest.fixture(params=["coupling_flow"])
def inference_network(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["inference_network", "summary_network"])
def network(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def resnet():
    from bayesflow.experimental.networks import ResNet
    return ResNet()


@pytest.fixture(params=[None])
def summary_network(request):
    if request.param is None:
        return None

    return request.getfixturevalue(request.param)
