import functools

import keras
import numpy as np
import pytest

from tests.utils import allclose


@pytest.mark.parametrize("automatic", [True, False])
def test_build(automatic, invertible_layer, random_input):
    assert invertible_layer.built is False

    if automatic:
        invertible_layer(random_input)
    else:
        invertible_layer.build(keras.ops.shape(random_input))

    assert invertible_layer.built is True


def test_variable_batch_size(invertible_layer, random_input):
    # manual build with one batch size
    invertible_layer.build(keras.ops.shape(random_input))

    # run with another batch size
    for _ in range(10):
        batch_size = np.random.randint(1, 10)
        new_input = keras.ops.zeros((batch_size,) + keras.ops.shape(random_input)[1:])
        invertible_layer(new_input)


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


def test_cycle_consistency(invertible_layer, random_input):
    # cycle-consistency means the forward and inverse methods are inverses of each other
    forward_output, forward_log_det = invertible_layer(random_input)
    inverse_output, inverse_log_det = invertible_layer(forward_output, inverse=True)

    assert allclose(random_input, inverse_output)
    assert allclose(forward_log_det, -inverse_log_det)


@pytest.mark.torch
def test_jacobian_numerically(invertible_layer, random_input):
    import torch

    forward_output, forward_log_det = invertible_layer(random_input)
    numerical_forward_jacobian, *_ = torch.autograd.functional.jacobian(invertible_layer, random_input, vectorize=True)

    # TODO: torch is somehow permuted wrt keras
    numerical_forward_log_det = [keras.ops.log(keras.ops.abs(keras.ops.det(numerical_forward_jacobian[i, :, i, :]))) for i in range(keras.ops.shape(random_input)[0])]
    numerical_forward_log_det = keras.ops.stack(numerical_forward_log_det, axis=0)

    assert allclose(forward_log_det, numerical_forward_log_det, rtol=1e-4, atol=1e-5)

    inverse_output, inverse_log_det = invertible_layer(random_input, inverse=True)

    numerical_inverse_jacobian, *_ = torch.autograd.functional.jacobian(functools.partial(invertible_layer, inverse=True), random_input, vectorize=True)

    # TODO: torch is somehow permuted wrt keras
    numerical_inverse_log_det = [keras.ops.log(keras.ops.abs(keras.ops.det(numerical_inverse_jacobian[i, :, i, :]))) for i in range(keras.ops.shape(random_input)[0])]
    numerical_inverse_log_det = keras.ops.stack(numerical_inverse_log_det, axis=0)

    assert allclose(inverse_log_det, numerical_inverse_log_det, rtol=1e-4, atol=1e-5)
