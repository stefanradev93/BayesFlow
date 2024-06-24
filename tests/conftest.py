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
