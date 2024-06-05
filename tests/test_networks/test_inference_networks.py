
import functools

import keras
import numpy as np
import pytest

from tests.utils import allclose, assert_models_equal


def test_build(inference_network, random_samples, random_conditions):
    assert inference_network.built is False

    inference_network(random_samples, conditions=random_conditions)

    assert inference_network.built is True

    # check the model has variables
    assert inference_network.variables, "Model has no variables."


def test_variable_batch_size(inference_network, random_samples, random_conditions):
    # build with one batch size
    inference_network(random_samples, conditions=random_conditions)

    # run with another batch size
    batch_sizes = np.random.choice(10, replace=False, size=3)
    for bs in batch_sizes:
        new_input = keras.ops.zeros((bs,) + keras.ops.shape(random_samples)[1:])
        if random_conditions is None:
            new_conditions = None
        else:
            new_conditions = keras.ops.zeros((bs,) + keras.ops.shape(random_conditions)[1:])

        inference_network(new_input, conditions=new_conditions)
        inference_network(new_input, conditions=new_conditions, inverse=True)


@pytest.mark.parametrize("jacobian", [True, False])
def test_output_structure(jacobian, inference_network, random_samples, random_conditions):
    output = inference_network(random_samples, conditions=random_conditions, jacobian=jacobian)

    if jacobian:
        assert isinstance(output, tuple)
        assert len(output) == 2

        forward_output, forward_log_det = output

        assert keras.ops.is_tensor(forward_output)
        assert keras.ops.is_tensor(forward_log_det)
    else:
        assert keras.ops.is_tensor(output)


def test_output_shape(inference_network, random_samples, random_conditions):
    forward_output, forward_log_det = inference_network(random_samples, conditions=random_conditions, jacobian=True)

    assert keras.ops.shape(forward_output) == keras.ops.shape(random_samples)
    assert keras.ops.shape(forward_log_det) == (keras.ops.shape(random_samples)[0],)

    inverse_output, inverse_log_det = inference_network(random_samples, conditions=random_conditions, jacobian=True, inverse=True)

    assert keras.ops.shape(inverse_output) == keras.ops.shape(random_samples)
    assert keras.ops.shape(inverse_log_det) == (keras.ops.shape(random_samples)[0],)


def test_cycle_consistency(inference_network, random_samples, random_conditions):
    # cycle-consistency means the forward and inverse methods are inverses of each other
    forward_output, forward_log_det = inference_network(random_samples, conditions=random_conditions, jacobian=True)
    inverse_output, inverse_log_det = inference_network(forward_output, conditions=random_conditions, jacobian=True, inverse=True)

    assert allclose(random_samples, inverse_output)
    assert allclose(forward_log_det, -inverse_log_det)


@pytest.mark.torch
def test_jacobian_numerically(inference_network, random_samples, random_conditions):
    import torch

    forward_output, forward_log_det = inference_network(random_samples, conditions=random_conditions, jacobian=True)

    def f(x):
        return inference_network(x, conditions=random_conditions)

    numerical_forward_jacobian, *_ = torch.autograd.functional.jacobian(f, random_samples, vectorize=True)

    # TODO: torch is somehow permuted wrt keras
    numerical_forward_log_det = [keras.ops.log(keras.ops.abs(keras.ops.det(numerical_forward_jacobian[:, i, :]))) for i in range(keras.ops.shape(random_samples)[0])]
    numerical_forward_log_det = keras.ops.stack(numerical_forward_log_det, axis=0)

    assert allclose(forward_log_det, numerical_forward_log_det, rtol=1e-4, atol=1e-5)

    inverse_output, inverse_log_det = inference_network(random_samples, conditions=random_conditions, jacobian=True, inverse=True)

    def f(x):
        return inference_network(x, conditions=random_conditions, inverse=True)

    numerical_inverse_jacobian, *_ = torch.autograd.functional.jacobian(f, random_samples, vectorize=True)

    # TODO: torch is somehow permuted wrt keras
    numerical_inverse_log_det = [keras.ops.log(keras.ops.abs(keras.ops.det(numerical_inverse_jacobian[:, i, :]))) for i in range(keras.ops.shape(random_samples)[0])]
    numerical_inverse_log_det = keras.ops.stack(numerical_inverse_log_det, axis=0)

    assert allclose(inverse_log_det, numerical_inverse_log_det, rtol=1e-4, atol=1e-5)


def test_serialize_deserialize(tmp_path, inference_network, random_samples, random_conditions):
    # to save, the model must be built
    inference_network(random_samples, conditions=random_conditions)

    keras.saving.save_model(inference_network, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(inference_network, loaded)
