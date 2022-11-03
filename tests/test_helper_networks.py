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
import tensorflow as tf

import pytest

from bayesflow.helper_networks import DenseCouplingNet, Permutation


@pytest.mark.parametrize("input_dim", [3, 16])
@pytest.mark.parametrize("meta", [
    dict(dense_args={'units': 32}, spec_norm=True, n_dense=1),
    dict(dense_args={'units': 64}, spec_norm=True, n_dense=2),
    dict(dense_args={'units': 32}, spec_norm=False, n_dense=1),
    dict(dense_args={'units': 64}, spec_norm=False, n_dense=2)
])
@pytest.mark.parametrize("condition", [False, True])
def test_dense_coupling_net(input_dim, meta, condition):
    """This function tests the fidelity of the `DenseCouplingNet` using different configurations."""

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

    # Shapes should be the same (i.e., bijection)
    assert out.shape == inp.shape


@pytest.mark.parametrize("input_dim", [3, 8])
@pytest.mark.parametrize("shape", ['2d', '3d'])
def test_permutation(input_dim, shape):
    """Tests the fixed `Permutation` layer in terms of invertibility and input-output fidelity."""

    # Create randomized input
    if shape == '2d':
        x = tf.random.normal((np.random.randint(low=1, high=32), input_dim))
    else:
        x = tf.random.normal((np.random.randint(low=1, high=32), np.random.randint(low=1, high=32), input_dim))

    # Create permutation layer and apply P_inv * P(input)
    perm = Permutation(input_dim)
    x_rec = perm(perm(x), inverse=True).numpy()

    # Inverse should recover input
    assert np.allclose(x, x_rec)
