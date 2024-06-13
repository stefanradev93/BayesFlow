import keras
import pytest

from bayesflow.experimental.configurators import Configurator

"""
TODO: with / without conditions

TODO: with / without summary network inputs and outputs

TODO: for data with any number of data dimensions

TODO: (optional) for data with any number of batch dimensions
"""

@pytest.fixture()
def random_data():
    return {
        'var1': keras.random.normal((5,5)),
        'var2': keras.random.normal((5,5)),
        'var3': keras.random.normal((5,5)),
        'summary_outputs': keras.random.normal((5,5)),
    }

@pytest.fixture()
def test_params():
    return {
        'inference_variables': ["var1"],
        'inference_conditions': ["var2, var3"],
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
def configurator_sparse(test_params):
    return Configurator(
        inference_variables=test_params['inference_variables'],
    )


# Test for correct construction of Configurator with all args
def test_configurator_init(test_params, configurator: Configurator):
    config = configurator
    assert config.inference_variables == test_params['inference_variables']
    assert config.inference_conditions == test_params['inference_conditions']
    assert config.summary_variables == test_params['summary_variables']
    assert config.summary_conditions == test_params['summary_conditions']


# Test for correct construction of Configurator with only inference_vars
def test_sparse_configurator_init(test_params, configurator_sparse: Configurator):
    config = configurator_sparse
    assert config.inference_variables == test_params['inference_variables']
    assert config.inference_conditions == []
    assert config.summary_variables == []
    assert config.summary_conditions == []

# TODO: Test successful configure_inference_variables (check shapes)

# TODO: Test successful configure_inference_conditions w/o summary_outputs in either

# TODO: Test successful configure_inference_conditions w/ summary_outputs in data, not in keys

# TODO: Test successful configure_inference_conditions w/ summary_outputs in both

# TODO: Test successful configure_summary_variables

# TODO: Test successful configure_summary_conditions

# TODO: Test for None return when keys == None for all params