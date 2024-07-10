import keras
import pytest


@pytest.fixture(params=[2, 3])
def batch_size(request):
    return request.param


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


@pytest.fixture(params=[2, 3])
def set_size(request):
    return request.param


@pytest.fixture(params=[])
def summary_network(request):
    return request.getfixturevalue(request.param)
