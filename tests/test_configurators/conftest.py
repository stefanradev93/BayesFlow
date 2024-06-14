import keras
import pytest
from bayesflow.experimental.configurators import Configurator


@pytest.fixture()
def test_shape():
    return (5,7)


@pytest.fixture()
def random_data(test_shape):
    return {
        'var1': keras.random.normal(test_shape),
        'var2': keras.random.normal(test_shape),
        'var3': keras.random.normal(test_shape),
        'summary_outputs': keras.random.normal(test_shape),
    }


@pytest.fixture()
def random_data_no_output(test_shape):
    return {
        'var1': keras.random.normal(test_shape),
        'var2': keras.random.normal(test_shape),
        'var3': keras.random.normal(test_shape),
    }


@pytest.fixture()
def test_params():
    return {
        'inference_variables': ["var1"],
        'inference_conditions': ["var2", "var3", "summary_outputs"],
        'summary_variables': ["var1"],
        'summary_conditions': ["var2"]
    }


@pytest.fixture()
def test_params_no_output():
    return {
        'inference_variables': ["var1"],
        'inference_conditions': ["var2", "var3"],
        'summary_variables': ["var1"],
        'summary_conditions': ["var2"]
    }


@pytest.fixture()
def configurator(test_params):
    return Configurator(
        inference_variables=test_params['inference_variables'],
        inference_conditions=test_params['inference_conditions'],
        summary_variables=test_params['summary_variables'],
        summary_conditions=test_params['summary_conditions']
    )


@pytest.fixture()
def configurator_no_output(test_params_no_output):
    return Configurator(
        inference_variables=test_params_no_output['inference_variables'],
        inference_conditions=test_params_no_output['inference_conditions'],
        summary_variables=test_params_no_output['summary_variables'],
        summary_conditions=test_params_no_output['summary_conditions']
    )


@pytest.fixture()
def configurator_sparse(test_params):
    return Configurator(
        inference_variables=test_params['inference_variables'],
    )