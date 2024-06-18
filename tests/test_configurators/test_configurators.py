from keras import ops
import pytest


def test_inference_vars_filter(random_data, configurator):
    # Tests for correct output shape when querying inference variables
    filtered_data = configurator.configure_inference_variables(random_data)
    expected = ops.concatenate([random_data[v] for v in configurator.inference_variables], axis=-1)
    assert filtered_data.shape == expected.shape


def test_inferences_conds_filter(random_data, configurator):
    # Tests for correct output shape when querying inference conditions w.r.t. summary_outputs
    if not configurator.inference_conditions:
        if "summary_outputs" in random_data:
            assert configurator.configure_inference_conditions(random_data).shape == random_data["summary_outputs"].shape
        else:
            assert configurator.configure_inference_conditions(random_data) is None
    elif not "summary_outputs" in random_data and "summary_outputs" in configurator.inference_conditions:
        with pytest.raises(KeyError):
            filtered_data = configurator.configure_inference_conditions(random_data)
    else:
        filtered_data = configurator.configure_inference_conditions(random_data)
        tensors = [random_data[v] for v in configurator.inference_conditions]
        if "summary_outputs" in random_data and not "summary_outputs" in configurator.inference_conditions:
            tensors.append(random_data["summary_outputs"])
        expected = ops.concatenate(tensors, axis=-1)
        assert filtered_data.shape == expected.shape


def test_summary_vars_filter(random_data, configurator):
    # Tests for correct output shape when querying summary variables
    if not configurator.summary_variables:
        assert configurator.configure_summary_variables(random_data) is None
    else:
        filtered_data = configurator.configure_summary_variables(random_data)
        expected = ops.concatenate([random_data[v] for v in configurator.summary_variables], axis=-1)
        assert filtered_data.shape == expected.shape


def test_summary_conds_filter(random_data, configurator):
    # Tests for correct output shape when querying summary conditions
    if not configurator.summary_conditions:
        assert configurator.configure_summary_conditions(random_data) is None
    else:
        filtered_data = configurator.configure_summary_conditions(random_data)
        expected = ops.concatenate([random_data[v] for v in configurator.summary_conditions], axis=-1)
        assert filtered_data.shape == expected.shape