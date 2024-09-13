import numpy as np
import pytest


def forward_transform(x):
    return x + 1


def inverse_transform(x):
    return x - 1


@pytest.fixture()
def custom_objects():
    return globals() | np.__dict__


@pytest.fixture()
def data_adapter():
    from bayesflow.data_adapters import ConcatenateKeysDataAdapter
    from bayesflow.data_adapters.transforms import LambdaTransform, Normalize

    return ConcatenateKeysDataAdapter(
        x=["x1", "x2"],
        y=["y1", "y2"],
        transforms=[
            Normalize("x1", mean=np.array([0.0]), std=np.array([1.0])),
            # use a lambda transform with global functions
            LambdaTransform("x2", forward_transform, inverse_transform),
        ],
    )


@pytest.fixture()
def random_data():
    return {
        "x1": np.random.standard_normal(size=(32, 1)).astype("float32"),
        "x2": np.random.standard_normal(size=(32, 1)).astype("float32"),
        "y1": np.random.standard_normal(size=(32, 2)).astype("float32"),
        "y2": np.random.standard_normal(size=(32, 2)).astype("float32"),
    }
