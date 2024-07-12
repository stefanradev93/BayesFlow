import keras
import pytest

from bayesflow.configurators import Configurator, DictConfigurator


@pytest.fixture(params=[2, 3])
def batch_size(request):
    return request.param


@pytest.fixture(params=[2, 3])
def set_size(request):
    return request.param


@pytest.fixture(params=[2, 3])
def num_features(request):
    return request.param


@pytest.fixture(params=[False, True])
def random_data(request, batch_size, set_size, num_features):
    data = {
        "var1": keras.random.normal((batch_size, set_size, num_features)),
        "var2": keras.random.normal((batch_size, set_size, num_features)),
        "var3": keras.random.normal((batch_size, set_size, num_features)),
        "summary_inputs": keras.random.normal((batch_size, set_size, num_features)),
    }
    if request.param:
        data["summary_outputs"] = keras.random.normal((batch_size, set_size, num_features))
    return data


@pytest.fixture(params=[False, True])
def test_params(request):
    args = {
        "inference_variables": ["var1"],
        "inference_conditions": ["var2", "var3"],
        "summary_variables": ["var1"],
    }
    if request.param:
        args["inference_conditions"].append("summary_outputs")
    return args


@pytest.fixture(params=[False, True])
def configurator(request, test_params):
    if request.param:
        return Configurator(inference_variables=test_params["inference_variables"])
    return Configurator(
        inference_variables=test_params["inference_variables"],
        inference_conditions=test_params["inference_conditions"],
        summary_variables=test_params["summary_variables"],
    )


@pytest.fixture()
def random_multisource_data(request, batch_size, set_size, num_features):
    data = {
        "var1": keras.random.normal((batch_size, set_size, num_features)),
        "var2": keras.random.normal((batch_size, set_size, num_features)),
        "var3": keras.random.normal((batch_size, 2 * set_size, num_features + 1)),
    }
    return data


@pytest.fixture()
def dict_configurator(request, test_params):
    return DictConfigurator(
        inference_variables=["var1"],
        inference_conditions=["var2"],
        summary_variables=["var2", "var3"],
    )
