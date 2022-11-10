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

from bayesflow.inference_networks import InvertibleNetwork
from bayesflow.helper_functions import build_meta_dict
from bayesflow.default_settings import DEFAULT_SETTING_INVERTIBLE_NET


@pytest.mark.parametrize("input_shape", ['2d', '3d'])
@pytest.mark.parametrize("condition", [True, False])
@pytest.mark.parametrize("use_act_norm", [True, False])
@pytest.mark.parametrize("use_soft_flow", [True, False])
@pytest.mark.parametrize("num_coupling_layers", [1, 8])
def test_invertible_network(input_shape, condition, use_act_norm, use_soft_flow, num_coupling_layers):
    """Tests the `InvertibleNetwork` core class using a couple of relevant configurations."""

    # Randomize units and input dim
    units_t = np.random.randint(low=2, high=32)
    units_s = np.random.randint(low=2, high=32)
    input_dim = np.random.randint(low=2, high=32)

    # Create settings dictionaries and network
    dense_net_settings = {
        't_args': {
            'dense_args': dict(units=units_t, kernel_initializer='glorot_uniform', activation='elu'),
            'num_dense': 1,
            'spec_norm': True
        },
        's_args': {
            'dense_args': dict(units=units_s, kernel_initializer='glorot_normal', activation='relu'),
            'num_dense': 2,
            'spec_norm': False
        },
    }
    settings = build_meta_dict(user_dict={
        'coupling_net_settings': dense_net_settings,
        'use_act_norm': use_act_norm,
        'use_soft_flow': use_soft_flow,
        'num_coupling_layers': num_coupling_layers,
        'num_params': input_dim
    },
    default_setting=DEFAULT_SETTING_INVERTIBLE_NET)

    network = InvertibleNetwork(**settings)

    # Create randomized input and output conditions
    batch_size = np.random.randint(low=1, high=32)
    if input_shape == '2d':
        inp = np.random.normal(size=(batch_size, input_dim)).astype(np.float32)
    else:
        n_obs = np.random.randint(low=1, high=32)
        inp = np.random.normal(size=(batch_size, n_obs, input_dim)).astype(np.float32)
    if condition :
        condition_dim = np.random.randint(low=1, high=32)
        condition = np.random.normal(size=(batch_size, condition_dim)).astype(np.float32)
    else:
        condition = None

    # Forward and inverse pass
    z, ldj = network(inp, condition)
    z = z.numpy()
    inp_rec = network(z, condition, inverse=True).numpy()

    # Test attributes
    assert network.latent_dim == input_dim
    assert len(network.coupling_layers) == num_coupling_layers
    for l in network.coupling_layers:
        assert l.permutation is not None
        if use_act_norm:
            assert l.act_norm is not None
        else:
            assert l.act_norm is None
    if use_soft_flow:
        assert network.soft_flow is True
    else:
        assert network.soft_flow is False
    # Test invertibility (in case no soft flow)
    if not use_soft_flow:
        assert np.allclose(inp, inp_rec, atol=1e-5)
    # Test shapes (bijectivity)
    assert z.shape == inp.shape
    assert z.shape[-1] == input_dim
    if input_shape == '2d':
        assert ldj.shape[0] == inp.shape[0]
    else:
        assert ldj.shape[0] == inp.shape[0] and ldj.shape[1] == inp.shape[1]
