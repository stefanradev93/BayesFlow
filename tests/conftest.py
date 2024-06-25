import keras
import pytest


BACKENDS = ["jax", "numpy", "tensorflow", "torch"]


def pytest_runtest_setup(item):
    """Skips backends by test markers. Unmarked tests are treated as backend-agnostic"""
    backend = keras.backend.backend()

    test_backends = [mark.name for mark in item.iter_markers() if mark.name in BACKENDS]

    if test_backends and backend not in test_backends:
        pytest.skip(f"Skipping backend '{backend}' for test {item}, which is registered for backends {test_backends}.")


@pytest.fixture(autouse=True, scope="function")
def random_seed():
    seed = 0
    keras.utils.set_random_seed(seed)
    return seed


@pytest.fixture()
def coupling_flow():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(depth=2, subnet_kwargs=dict(depth=2, width=64))


@pytest.fixture()
def flow_matching():
    from bayesflow.networks import FlowMatching

    return FlowMatching(network_kwargs=dict(depth=2, width=64))


@pytest.fixture(params=["coupling_flow", "flow_matching"])
def inference_network(request):
    return request.getfixturevalue(request.param)
