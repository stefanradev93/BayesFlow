
import functools

import keras
import numpy as np
import pytest

from tests.utils import assert_layers_equal


@pytest.mark.parametrize("automatic", [True, False])
def test_build(automatic, inference_network, random_samples):
    assert inference_network.built is False

    if automatic:
        inference_network(random_samples)
    else:
        inference_network.build(keras.ops.shape(random_samples))

    assert inference_network.built is True

    # check the model has variables
    assert inference_network.variables, "Model has no variables."


def test_variable_batch_size(inference_network, random_samples):
    # build with one batch size
    inference_network.build(keras.ops.shape(random_samples))

    # run with another batch size
    for _ in range(10):
        batch_size = np.random.randint(1, 10)
        new_input = keras.ops.zeros((batch_size,) + keras.ops.shape(random_samples)[1:])
        inference_network(new_input)
        inference_network(new_input, inverse=True)


def test_output_structure(invertible_layer, random_input):
    output = invertible_layer(random_input)

    assert isinstance(output, tuple)
    assert len(output) == 2

    forward_output, forward_log_det = output

    assert keras.ops.is_tensor(forward_output)
    assert keras.ops.is_tensor(forward_log_det)


def test_output_shape(invertible_layer, random_input):
    forward_output, forward_log_det = invertible_layer(random_input)

    assert keras.ops.shape(forward_output) == keras.ops.shape(random_input)
    assert keras.ops.shape(forward_log_det) == (keras.ops.shape(random_input)[0],)

    inverse_output, inverse_log_det = invertible_layer(random_input, inverse=True)

    assert keras.ops.shape(inverse_output) == keras.ops.shape(random_input)
    assert keras.ops.shape(inverse_log_det) == (keras.ops.shape(random_input)[0],)


def test_cycle_consistency(inference_network, random_samples):
    # cycle-consistency means the forward and inverse methods are inverses of each other
    forward_output, forward_log_det = inference_network(random_samples, jacobian=True)
    inverse_output, inverse_log_det = inference_network(forward_output, inverse=True, jacobian=True)

    assert keras.ops.all(keras.ops.isclose(random_samples, inverse_output))
    assert keras.ops.all(keras.ops.isclose(forward_log_det, -inverse_log_det))


@pytest.mark.torch
def test_jacobian_numerically(invertible_layer, random_input):
    import torch

    forward_output, forward_log_det = invertible_layer(random_input, jacobian=True)
    numerical_forward_jacobian, _ = torch.autograd.functional.jacobian(invertible_layer, random_input, vectorize=True)

    # TODO: torch is somehow permuted wrt keras
    numerical_forward_log_det = [keras.ops.log(keras.ops.abs(keras.ops.det(numerical_forward_jacobian[i, :, i, :]))) for i in range(keras.ops.shape(random_input)[0])]
    numerical_forward_log_det = keras.ops.stack(numerical_forward_log_det, axis=0)

    assert keras.ops.all(keras.ops.isclose(forward_log_det, numerical_forward_log_det))

    inverse_output, inverse_log_det = invertible_layer(random_input, inverse=True, jacobian=True)

    numerical_inverse_jacobian, _ = torch.autograd.functional.jacobian(functools.partial(invertible_layer, inverse=True), random_input, vectorize=True)

    # TODO: torch is somehow permuted wrt keras
    numerical_inverse_log_det = [keras.ops.log(keras.ops.abs(keras.ops.det(numerical_inverse_jacobian[i, :, i, :]))) for i in range(keras.ops.shape(random_input)[0])]
    numerical_inverse_log_det = keras.ops.stack(numerical_inverse_log_det, axis=0)

    assert keras.ops.all(keras.ops.isclose(inverse_log_det, numerical_inverse_log_det))


def test_serialize_deserialize(tmp_path, inference_network, random_samples):
    inference_network.build(keras.ops.shape(random_samples))

    keras.saving.save_model(inference_network, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(inference_network, loaded)
