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

from bayesflow.coupling_networks import AffineCoupling, CouplingLayer, SplineCoupling
from bayesflow.helper_networks import Orthogonal, Permutation


@pytest.mark.parametrize("condition", [True, False])
@pytest.mark.parametrize("coupling_design", ["affine", "spline"])
@pytest.mark.parametrize("permutation", ["fixed", "learnable"])
@pytest.mark.parametrize("use_act_norm", [True, False])
@pytest.mark.parametrize("input_shape", ["2d", "3d"])
def test_coupling_layer(condition, coupling_design, permutation, use_act_norm, input_shape):
    """Tests the ``CouplingLayer`` instance with various configurations."""

    # Randomize units and input dim
    units = np.random.randint(low=2, high=32)
    input_dim = np.random.randint(low=2, high=32)

    # Create settings dictionaries and network
    if coupling_design == "affine":
        coupling_settings = {
            "dense_args": dict(units=units, activation="elu"),
            "num_dense": 1,
        }
    else:
        coupling_settings = {"dense_args": dict(units=units, activation="elu"), "num_dense": 1, "bins": 8}
    settings = {
        "latent_dim": input_dim,
        "coupling_settings": coupling_settings,
        "permutation": permutation,
        "use_act_norm": use_act_norm,
        "coupling_design": coupling_design,
    }

    network = CouplingLayer(**settings)

    # Create randomized input and output conditions
    batch_size = np.random.randint(low=1, high=32)
    if input_shape == "2d":
        inp = np.random.normal(size=(batch_size, input_dim)).astype(np.float32)
    else:
        n_obs = np.random.randint(low=1, high=32)
        inp = np.random.normal(size=(batch_size, n_obs, input_dim)).astype(np.float32)
    if condition:
        condition_dim = np.random.randint(low=1, high=32)
        condition = np.random.normal(size=(batch_size, condition_dim)).astype(np.float32)
    else:
        condition = None

    # Forward and inverse pass
    z, ldj = network(inp, condition)
    z = z.numpy()
    inp_rec = network(z, condition, inverse=True).numpy()

    # Test attributes
    if permutation == "fixed":
        assert not network.permutation.trainable
        assert isinstance(network.permutation, Permutation)
    else:
        assert isinstance(network.permutation, Orthogonal)
        assert network.permutation.trainable
    if use_act_norm:
        assert network.act_norm is not None
    else:
        assert network.act_norm is None

    # Test coupling type
    if coupling_design == "affine":
        assert isinstance(network.net1, AffineCoupling) and isinstance(network.net2, AffineCoupling)
    elif coupling_design == "spline":
        assert isinstance(network.net1, SplineCoupling) and isinstance(network.net2, SplineCoupling)

    # Test invertibility
    assert np.allclose(inp, inp_rec, atol=1e-5)
    # Test shapes (bijectivity)
    assert z.shape == inp.shape
    if input_shape == "2d":
        assert ldj.shape[0] == inp.shape[0]
    else:
        assert ldj.shape[0] == inp.shape[0] and ldj.shape[1] == inp.shape[1]
