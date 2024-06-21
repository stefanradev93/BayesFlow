import keras
import pytest


BACKENDS = ["jax", "numpy", "tensorflow", "torch"]


def pytest_runtest_setup(item):
    """Skips backends by test markers. Unmarked tests are treated as backend-agnostic"""
    backend = keras.backend.backend()

    test_backends = [mark.name for mark in item.iter_markers() if mark.name in BACKENDS]

    if test_backends and backend not in test_backends:
        pytest.skip(
            f"Skipping backend '{backend}' for test {item}, " f"which is registered for backends {test_backends}."
        )
