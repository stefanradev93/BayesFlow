
import pytest

import bayesflow.experimental as bf


@pytest.fixture()
def resnet():
    return bf.networks.ResNet.new()


@pytest.fixture()
def coupling_flow():
    return bf.networks.CouplingFlow.new()


@pytest.fixture(params=["resnet", "coupling_flow"])
def network(request):
    return request.getfixturevalue(request.param)
