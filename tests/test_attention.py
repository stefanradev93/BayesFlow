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

from bayesflow.attention import (
    InducedSelfAttentionBlock,
    MultiHeadAttentionBlock,
    PoolingWithAttention,
    SelfAttentionBlock,
)


def _get_attention_and_dense_kwargs():
    """Helper function for generating fixed hyperparameter settings."""

    attention = dict(num_heads=2, key_dim=32)
    dense = dict(units=8, activation="relu")
    return attention, dense


def _get_randomized_3d_data(batch_size=None, num_obs=None, num_dims=None):
    """Helper function to generate 3d input arrays with randomized shapes."""

    if batch_size is None:
        batch_size = np.random.randint(low=1, high=5)
    if num_obs is None:
        num_obs = np.random.randint(low=2, high=10)
    if num_dims is None:
        num_dims = np.random.randint(low=2, high=8)
    inp = np.random.normal(size=(batch_size, num_obs, num_dims)).astype(np.float32)
    return inp


@pytest.mark.parametrize("use_layer_norm", [True, False])
@pytest.mark.parametrize("num_dense_fc", [1, 2])
def test_multihead_attention_block(use_layer_norm, num_dense_fc):
    """Tests the ``MultiHeadAttentionBlock`` (MAB) block with relevant configurations."""

    # Create synthetic 3d data
    inp1 = _get_randomized_3d_data()
    inp2 = _get_randomized_3d_data(batch_size=inp1.shape[0])

    # Get fixed settings for attention and internal fc
    att_dict, dense_dict = _get_attention_and_dense_kwargs()

    # Create instance and apply to synthetic input
    block = MultiHeadAttentionBlock(inp1.shape[2], att_dict, num_dense_fc, dense_dict, use_layer_norm)
    out = block(inp1, inp2)

    # Ensuse shape conditions
    assert out.shape[0] == inp1.shape[0]
    assert out.shape[1] == inp1.shape[1]
    assert out.shape[2] == inp1.shape[2]

    # Ensure attribute integrity
    if use_layer_norm:
        assert block.ln_pre is not None and block.ln_post is not None
    else:
        assert block.ln_pre is None and block.ln_post is None
    assert len(block.fc.layers) == num_dense_fc + 1


def test_self_attention_block(num_dense_fc=1, use_layer_norm=True):
    """Tests the ``SelfAttentionBlock`` (SAB) block with relevant configurations."""

    # Create synthetic 3d data
    inp = _get_randomized_3d_data()

    # Get fixed settings for attention and internal fc
    att_dict, dense_dict = _get_attention_and_dense_kwargs()

    # Create instance and apply to synthetic input
    block = SelfAttentionBlock(inp.shape[2], att_dict, num_dense_fc, dense_dict, use_layer_norm)
    out = block(inp)

    # Ensuse shape conditions
    assert out.shape[0] == inp.shape[0]
    assert out.shape[1] == inp.shape[1]
    assert out.shape[2] == inp.shape[2]

    # Ensure attribute integrity
    assert block.mab.ln_pre is not None and block.mab.ln_post is not None
    assert len(block.mab.fc.layers) == 2

    # Ensure permutation equivariance
    perm = np.random.permutation(inp.shape[1])
    perm_inp = inp[:, perm, :]
    perm_out = block(perm_inp)
    assert np.allclose(tf.reduce_sum(perm_out, axis=1), tf.reduce_sum(out, axis=1), atol=1e-5)


@pytest.mark.parametrize("num_inducing_points", [4, 8])
def test_induced_self_attention_block(num_inducing_points, num_dense_fc=1, use_layer_norm=False):
    """Tests the ``IncudedSelfAttentionBlock`` (ISAB) block with relevant configurations."""

    # Create synthetic 3d data
    inp = _get_randomized_3d_data()

    # Get fixed settings for attention and internal fc
    att_dict, dense_dict = _get_attention_and_dense_kwargs()

    # Create instance and apply to synthetic input
    block = InducedSelfAttentionBlock(
        inp.shape[2], att_dict, num_dense_fc, dense_dict, use_layer_norm, num_inducing_points
    )
    out = block(inp)

    # Ensure permutation equivariance
    perm = np.random.permutation(inp.shape[1])
    perm_inp = inp[:, perm, :]
    perm_out = block(perm_inp)
    assert np.allclose(tf.reduce_sum(perm_out, axis=1), tf.reduce_sum(out, axis=1), atol=1e-5)


@pytest.mark.parametrize("summary_dim", [3, 8])
@pytest.mark.parametrize("num_seeds", [1, 3])
def test_pooling_with_attention(summary_dim, num_seeds, num_dense_fc=1, use_layer_norm=True):
    """Tests the ``PoolingWithAttention`` block using relevant configurations."""

    # Create synthetic 3d data
    inp = _get_randomized_3d_data()

    # Get fixed settings for attention and internal fc
    att_dict, dense_dict = _get_attention_and_dense_kwargs()

    # Create pooler and apply it to synthetic data
    pma = PoolingWithAttention(summary_dim, att_dict, num_dense_fc, dense_dict, use_layer_norm, num_seeds)
    out = pma(inp)

    # Ensure shape integrity
    assert len(out.shape) == 2
    assert out.shape[0] == inp.shape[0]
    assert out.shape[1] == int(num_seeds * summary_dim)

    # Ensure permutation invariance
    perm = np.random.permutation(inp.shape[1])
    perm_inp = inp[:, perm, :]
    perm_out = pma(perm_inp)
    assert np.allclose(out, perm_out, atol=1e-5)
