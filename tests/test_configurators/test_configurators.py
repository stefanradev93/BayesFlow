import keras
import pytest


def test_inference_variables_shape(random_data, configurator):
    # Tests for correct output shape when querying inference variables
    filtered_data = configurator.configure_inference_variables(random_data)
    expected = keras.ops.concatenate([random_data[v] for v in configurator.inference_variables], axis=-1)
    assert filtered_data.shape == expected.shape


def test_inference_conditions_shape(random_data, configurator):
    # Tests for correct output shape when querying inference conditions w.r.t. summary_outputs
    if not configurator.inference_conditions:
        if "summary_outputs" in random_data:
            assert (
                configurator.configure_inference_conditions(random_data).shape == random_data["summary_outputs"].shape
            )
        else:
            assert configurator.configure_inference_conditions(random_data) is None
    elif "summary_outputs" not in random_data and "summary_outputs" in configurator.inference_conditions:
        with pytest.raises(KeyError):
            filtered_data = configurator.configure_inference_conditions(random_data)
    else:
        filtered_data = configurator.configure_inference_conditions(random_data)
        tensors = [random_data[v] for v in configurator.inference_conditions]
        if "summary_outputs" in random_data and "summary_outputs" not in configurator.inference_conditions:
            tensors.append(random_data["summary_outputs"])
        expected = keras.ops.concatenate(tensors, axis=-1)
        assert filtered_data.shape == expected.shape


def test_summary_variables_shape(random_data, configurator):
    # Tests for correct output shape when querying summary variables
    if not configurator.summary_variables:
        assert configurator.configure_summary_variables(random_data) is None
    else:
        filtered_data = configurator.configure_summary_variables(random_data)
        expected = keras.ops.concatenate([random_data[v] for v in configurator.summary_variables], axis=-1)
        assert filtered_data.shape == expected.shape


def test_dict_summary_variables_shape(random_multisource_data, dict_configurator):
    # Tests for correct output shape when querying summary variables as dictionaries (e.g., for fusion summary nets)
    filtered_data = dict_configurator.configure_summary_variables(random_multisource_data)
    expected_len = len(dict_configurator.summary_variables)

    assert isinstance(filtered_data, dict) and len(filtered_data) == expected_len
