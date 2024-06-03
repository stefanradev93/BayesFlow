
import pytest

import keras


@pytest.fixture(params=[1, 2, 3])
def batch_size(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def num_features(request):
    return request.param


@pytest.fixture()
def random_samples(batch_size, num_features):
    return keras.random.normal((batch_size, num_features))

@pytest.fixture()
def diagonal_normal():
    from bayesflow.experimental.distributions import DiagonalNormal
    return DiagonalNormal()


@pytest.fixture(params=["diagonal_normal"])
def distribution(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def shape(batch_size, num_features):
    return batch_size, num_features