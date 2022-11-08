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

from bayesflow.summary_networks import InvariantModule, EquivariantModule, InvariantNetwork, MultiConv1D, MultiConvNetwork
from bayesflow.helper_functions import build_meta_dict
from bayesflow.default_settings import DEFAULT_SETTING_MULTI_CONV_NET, DEFAULT_SETTING_INVARIANT_NET


def _gen_randomized_3d_data(low=1, high=32, dtype=np.float32):
    """Helper function to generate randomized 3d data for summary modules, min and
    max dimensions for each axis are given by `low` and `high`."""

    # Randomize batch data
    x = np.random.default_rng().normal(size=(
        np.random.randint(low=low, high=high+1), 
        np.random.randint(low=low, high=high+1), 
        np.random.randint(low=low, high=high+1))
    ).astype(dtype)

    # Random permutation along first axis
    perm = np.random.default_rng().permutation(x.shape[1])
    x_perm = x[:, perm, :]
    return x, x_perm, perm


@pytest.mark.parametrize("n_dense_s1", [1, 2])
@pytest.mark.parametrize("n_dense_s2", [1, 2])
@pytest.mark.parametrize("output_dim", [3, 10])
def test_invariant_module(n_dense_s1, n_dense_s2, output_dim):
    """This function tests the permutation invariance property of the `InvariantModule` as well as
    its input-output integrity."""
    
    # Prepare settings for invariant module and create it
    meta = {
        'dense_s1_args': dict(units=8, activation='elu'),
        'dense_s2_args': dict(units=output_dim, activation='relu'),
        'n_dense_s1': n_dense_s1,
        'n_dense_s2': n_dense_s2
    }
    inv_module = InvariantModule(meta)

    # Create input and permuted version with randomized shapes 
    x, x_perm, _ = _gen_randomized_3d_data()

    # Pass unpermuted and permuted inputs
    out = inv_module(x).numpy()
    out_perm = inv_module(x_perm).numpy()

    # Assert outputs equal
    assert np.allclose(out, out_perm, atol=1e-6)
    # Assert shape 2d
    assert len(out.shape) == 2 and len(out_perm.shape) == 2
    # Assert first and last dimension equals output dimension
    assert x.shape[0] == out.shape[0] and x_perm.shape[0] == out.shape[0]
    assert out.shape[1] == output_dim and out_perm.shape[1] == output_dim


@pytest.mark.parametrize("n_dense_s3", [1, 2])
@pytest.mark.parametrize("output_dim", [3, 10])
def test_equivariant_module(n_dense_s3, output_dim):
    """This function tests the permutation equivariance property of the `EquivariantModule` as well
    as its input-output integrity."""

    # Prepare settings for equivariant module and create it
    meta = {
        'dense_s1_args': dict(units=8, activation='elu'),
        'dense_s2_args': dict(units=2, activation='relu'),
        'dense_s3_args': dict(units=output_dim),
        'n_dense_s1': 1,
        'n_dense_s2': 1,
        'n_dense_s3': n_dense_s3
    }
    equiv_module = EquivariantModule(meta)

    # Create input and permuted version with randomized shapes 
    x, x_perm, perm = _gen_randomized_3d_data()

    # Pass unpermuted and permuted inputs
    out = equiv_module(x).numpy()
    out_perm = equiv_module(x_perm).numpy()

    # Assert outputs equal
    assert np.allclose(out[:, perm, :], out_perm, atol=1e-6)
    # Assert shape 3d
    assert len(out.shape) == 3 and len(out_perm.shape) == 3
    # Assert first and last dimension equals output dimension
    assert x.shape[0] == out.shape[0] and x_perm.shape[0] == out.shape[0]
    assert out.shape[2] == output_dim and out_perm.shape[2] == output_dim


@pytest.mark.parametrize("n_equiv", [1, 3])
@pytest.mark.parametrize("summary_dim", [13, 10])
def test_invariant_network(n_equiv, summary_dim):
    """This function tests the fidelity of the invariant network with a couple of relevant
    configurations w.r.t. permutation invariance and output dimensions."""

    # Prepare settings for invariant network
    meta = build_meta_dict({
        'n_equiv': n_equiv,
        'summary_dim': summary_dim},
        default_setting=DEFAULT_SETTING_INVARIANT_NET
    )
    inv_net = InvariantNetwork(meta)

    # Create input and permuted version with randomized shapes 
    x, x_perm, _ = _gen_randomized_3d_data()

    # Pass unpermuted and permuted inputs
    out = inv_net(x).numpy()
    out_perm = inv_net(x_perm).numpy()

    # Assert outputs equal
    assert np.allclose(out, out_perm, atol=1e-6)
    # Assert shape 2d
    assert len(out.shape) == 2 and len(out_perm.shape) == 2
    # Assert batch and last dimension equals output dimension
    assert x.shape[0] == out.shape[0] and x_perm.shape[0] == out.shape[0]
    assert out.shape[1] == summary_dim and out_perm.shape[1] == summary_dim


@pytest.mark.parametrize("filters", [16, 32])
@pytest.mark.parametrize("max_kernel_size", [2, 6])
def test_multi_conv1d(filters, max_kernel_size):
    """This function tests the fidelity of the `MultiConv1D` module w.r.t. output dimensions
    using a number of relevant configurations."""

    # Create settings and network
    meta = {
        'layer_args': {
            'activation': 'relu',
            'filters': filters,
            'strides': 1,
            'padding': 'causal'
        },
        'min_kernel_size': 1,
        'max_kernel_size': max_kernel_size
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

@pytest.mark.parametrize("n_conv_layers", [1, 3])
@pytest.mark.parametrize("lstm_units", [16, 32])
def test_multi_conv_network(n_conv_layers, lstm_units):
    """This function tests the fidelity of the `MultiConvNetwork` w.r.t. output dimensions
    using a number of relevant configurations."""

    # Create settings dict and network
    meta = build_meta_dict({
        'n_conv_layers': n_conv_layers,
        'lstm_args': dict(units=lstm_units)},
        default_setting=DEFAULT_SETTING_MULTI_CONV_NET
    )
    net = MultiConvNetwork(meta)

    # Create test data and pass through network
    x, _, _ = _gen_randomized_3d_data()
    out = net(x)

    # Test shape 2d
    assert len(out.shape) == 2
    # Test summary stats equal default
    assert out.shape[1] == DEFAULT_SETTING_MULTI_CONV_NET.meta_dict['summary_dim']
    # Test first dimension unaltered
    assert out.shape[0] == x.shape[0]
