# Copyright (c) 2022 The BayesFlow Developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pytest
import tensorflow as tf

from bayesflow.helper_networks import *


def _gen_randomized_3d_data(low=1, high=32, dtype=np.float32):
    """Helper function to generate randomized 3d data for summary modules, min and
    max dimensions for each axis are given by ``low`` and ``high``."""

    # Randomize batch data
    x = (
        np.random.default_rng()
        .normal(
            size=(
                np.random.randint(low=low, high=high + 1),
                np.random.randint(low=low, high=high + 1),
                np.random.randint(low=low, high=high + 1),
            )
        )
        .astype(dtype)
    )

    # Random permutation along first axis
    perm = np.random.default_rng().permutation(x.shape[1])
    x_perm = x[:, perm, :]
    return x, x_perm, perm


@pytest.mark.parametrize("input_dim", [3, 16])
@pytest.mark.parametrize(
    "meta",
    [
        dict(
            dense_args={"units": 32, "activation": "elu"}, spec_norm=True, dropout=True, dropout_prob=0.2, num_dense=1
        ),
        dict(dense_args={"units": 64}, spec_norm=True, mc_dropout=True, dropout_prob=0.1, num_dense=2, residual=True),
        dict(dense_args={"units": 32, "activation": "relu"}, spec_norm=False, num_dense=1),
        dict(dense_args={"units": 64}, spec_norm=False, num_dense=2, residual=True),
    ],
)
@pytest.mark.parametrize("condition", [False, True])
def test_dense_coupling_net(input_dim, meta, condition):
    """Tests the fidelity of the ``DenseCouplingNet`` using different configurations."""

    # Randomize input batch to avoid a single fixed batch size
    B = np.random.randint(low=1, high=32)
    inp = tf.random.normal((B, input_dim))

    # Randomize condition
    if condition:
        cond_inp = tf.random.normal((B, np.random.randint(low=1, high=32)))
    else:
        cond_inp = None

    # Create dense coupling and apply to input
    dense = DenseCouplingNet(meta, input_dim)
    out = dense(inp, cond_inp).numpy()

    # Test number of layers
    # 2 Hidden + 2 Dropouts
    if meta.get("dropout") or meta.get("mc_dropout"):
        assert len(dense.fc.layers) == 2 * meta["num_dense"] + 1

    # 2 Hidden + 1 Out
    else:
        assert len(dense.fc.layers) == meta["num_dense"] + 1
    assert out.shape == inp.shape

    # Test residual
    if meta.get("residual"):
        assert dense.residual_output is not None
    else:
        assert dense.residual_output is None


@pytest.mark.parametrize("input_dim", [3, 8])
@pytest.mark.parametrize("shape", ["2d", "3d"])
def test_permutation(input_dim, shape):
    """Tests the fixed ``Permutation`` layer in terms of invertibility and input-output fidelity."""

    # Create randomized input
    if shape == "2d":
        x = tf.random.normal((np.random.randint(low=1, high=32), input_dim))
    else:
        x = tf.random.normal((np.random.randint(low=1, high=32), np.random.randint(low=1, high=32), input_dim))

    # Create permutation layer and apply P_inv * P(input)
    perm = Permutation(input_dim)
    x_rec = perm(perm(x), inverse=True).numpy()

    # Inverse should recover input regardless of input dim or shape
    assert np.allclose(x, x_rec)


@pytest.mark.parametrize("input_dim", [3, 8])
@pytest.mark.parametrize("shape", ["2d", "3d"])
def test_orthogonal(input_dim, shape):
    """Tests the learnable ``Orthogonal`` layer in terms of invertibility and input-output fidelity."""

    # Create randomized input
    if shape == "2d":
        x = tf.random.normal((np.random.randint(low=1, high=32), input_dim))
    else:
        x = tf.random.normal((np.random.randint(low=1, high=32), np.random.randint(low=1, high=32), input_dim))

    # Create permutation layer and apply P_inv * P(input)
    perm = Orthogonal(input_dim)
    x_rec = perm(perm(x)[0], inverse=True).numpy()

    # Inverse should recover input regardless of input dim or shape
    assert np.allclose(x, x_rec, atol=1e-5)


@pytest.mark.parametrize("input_dim", [2, 5])
@pytest.mark.parametrize("shape", ["2d", "3d"])
def test_actnorm(input_dim, shape):
    """Tests the ``ActNorm`` layer in terms of invertibility and shape integrity."""

    # Create randomized input
    if shape == "2d":
        x = tf.random.normal((np.random.randint(low=1, high=32), input_dim))
    else:
        x = tf.random.normal((np.random.randint(low=1, high=32), np.random.randint(low=1, high=32), input_dim))

    # Create ActNorm layer
    actnorm = ActNorm(latent_dim=input_dim, act_norm_init=None)

    # Forward - inverse
    z, _ = actnorm(x)
    x_rec = actnorm(z, inverse=True).numpy()

    # Inverse should recover input regardless of input dim or shape
    assert np.allclose(x, x_rec)


@pytest.mark.parametrize("n_dense_s1", [1, 2])
@pytest.mark.parametrize("n_dense_s2", [1, 2])
@pytest.mark.parametrize("output_dim", [3, 10])
def test_invariant_module(n_dense_s1, n_dense_s2, output_dim):
    """This function tests the permutation invariance property of the ``InvariantModule`` as well as
    its input-output integrity."""

    # Prepare settings for invariant module and create it
    meta = {
        "dense_s1_args": dict(units=8, activation="elu"),
        "dense_s2_args": dict(units=output_dim, activation="relu"),
        "num_dense_s1": n_dense_s1,
        "num_dense_s2": n_dense_s2,
        "pooling_fun": "mean",
    }
    inv_module = InvariantModule(meta)

    # Create input and permuted version with randomized shapes
    x, x_perm, _ = _gen_randomized_3d_data()

    # Pass unpermuted and permuted inputs
    out = inv_module(x).numpy()
    out_perm = inv_module(x_perm).numpy()

    # Assert outputs equal
    assert np.allclose(out, out_perm, atol=1e-5)
    # Assert shape 2d
    assert len(out.shape) == 2 and len(out_perm.shape) == 2
    # Assert first and last dimension equals output dimension
    assert x.shape[0] == out.shape[0] and x_perm.shape[0] == out.shape[0]
    assert out.shape[1] == output_dim and out_perm.shape[1] == output_dim


@pytest.mark.parametrize("n_dense_s3", [1, 2])
@pytest.mark.parametrize("output_dim", [3, 10])
def test_equivariant_module(n_dense_s3, output_dim):
    """This function tests the permutation equivariance property of the ``EquivariantModule`` as well
    as its input-output integrity."""

    # Prepare settings for equivariant module and create it
    meta = {
        "dense_s1_args": dict(units=8, activation="elu"),
        "dense_s2_args": dict(units=2, activation="relu"),
        "dense_s3_args": dict(units=output_dim),
        "num_dense_s1": 1,
        "num_dense_s2": 1,
        "num_dense_s3": n_dense_s3,
        "pooling_fun": "max",
    }
    equiv_module = EquivariantModule(meta)

    # Create input and permuted version with randomized shapes
    x, x_perm, perm = _gen_randomized_3d_data()

    # Pass unpermuted and permuted inputs
    out = equiv_module(x).numpy()
    out_perm = equiv_module(x_perm).numpy()

    # Assert outputs equal
    assert np.allclose(out[:, perm, :], out_perm, atol=1e-5)
    # Assert shape 3d
    assert len(out.shape) == 3 and len(out_perm.shape) == 3
    # Assert first and last dimension equals output dimension
    assert x.shape[0] == out.shape[0] and x_perm.shape[0] == out.shape[0]
    assert out.shape[2] == output_dim and out_perm.shape[2] == output_dim


@pytest.mark.parametrize("filters", [16, 32])
@pytest.mark.parametrize("max_kernel_size", [2, 6])
def test_multi_conv1d(filters, max_kernel_size):
    """This function tests the fidelity of the ``MultiConv1D`` module w.r.t. output dimensions
    using a number of relevant configurations."""

    # Create settings and network
    meta = {
        "layer_args": {"activation": "relu", "filters": filters, "strides": 1, "padding": "causal"},
        "min_kernel_size": 1,
        "max_kernel_size": max_kernel_size,
    }
    conv = MultiConv1D(meta)

    # Create test data and pass through network
    x, _, _ = _gen_randomized_3d_data()
    out = conv(x)

    # Assert shape 3d
    assert len(out.shape) == 3

    # Assert first and second axes equal
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == x.shape[1]

    # Assert number of channels as specified
    assert out.shape[2] == filters
