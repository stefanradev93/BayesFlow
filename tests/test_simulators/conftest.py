
import pytest


@pytest.fixture()
def two_moons():
    from bayesflow.simulators import TwoMoonsSimulator
    return TwoMoonsSimulator()


@pytest.fixture(params=["two_moons"])
def simulator(request):
    return request.getfixturevalue(request.param)
