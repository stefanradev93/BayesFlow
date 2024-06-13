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


# Test successful configure_inference_variables()
def test_inference_vars_filter(random_data, configurator: Configurator, test_shape):
    config = configurator
    filtered_data = config.configure_inference_variables(random_data)
    assert filtered_data.shape == test_shape


# Test successful configure_inference_conditions w/o summary_outputs in either
def test_inferences_conds_filter_no_outputs(random_data_no_output, configurator_no_output: Configurator, test_shape):
    config = configurator_no_output
    filtered_data = config.configure_inference_conditions(random_data_no_output)
    assert filtered_data.shape == (test_shape[0], test_shape[1] * 2)


# Test successful configure_inference_conditions w/ summary_outputs in data, not in keys
def test_inferences_conds_filter_partial_outputs(random_data, configurator_no_output: Configurator, test_shape):
    config = configurator_no_output
    filtered_data = config.configure_inference_conditions(random_data)
    assert filtered_data.shape == (test_shape[0], test_shape[1] * 3)


# Test successful configure_inference_conditions w/ summary_outputs in both
def test_inferences_conds_filter_with_outputs(random_data, configurator: Configurator, test_shape):
    config = configurator
    filtered_data = config.configure_inference_conditions(random_data)
    assert filtered_data.shape == (test_shape[0], test_shape[1] * 3)


# Test successful configure_summary_variables()
def test_summary_vars_filter(random_data, configurator: Configurator, test_shape):
    config = configurator
    filtered_data = config.configure_summary_variables(random_data)
    assert filtered_data.shape == test_shape


# Test successful configure_summary_conditions()
def test_summary_conds_filter(random_data, configurator: Configurator, test_shape):
    config = configurator
    filtered_data = config.configure_summary_conditions(random_data)
    assert filtered_data.shape == test_shape


# Test return None for filters when configuring sparse Configurator
def test_null_vars_and_conds(random_data_no_output, configurator_sparse: Configurator):
    config = configurator_sparse
    filtered_inference_conds = config.configure_inference_conditions(random_data_no_output)
    filtered_summary_vars = config.configure_summary_variables(random_data_no_output)
    filtered_summary_conds = config.configure_summary_conditions(random_data_no_output)
    assert filtered_inference_conds == None
    assert filtered_summary_vars == None
    assert filtered_summary_conds == None