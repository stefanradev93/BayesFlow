import numpy as np
import pytest


@pytest.fixture()
def data_adapter():
    from bayesflow.data_adapters import ConcatenateKeysDataAdapter

    return ConcatenateKeysDataAdapter(
        x=["x1", "x2"],
        y=["y1", "y2"],
    )


@pytest.fixture()
def random_data():
    return {
        "x1": np.random.standard_normal(size=(32, 1)).astype("float32"),
        "x2": np.random.standard_normal(size=(32, 1)).astype("float32"),
        "y1": np.random.standard_normal(size=(32, 2)).astype("float32"),
        "y2": np.random.standard_normal(size=(32, 2)).astype("float32"),
    }
