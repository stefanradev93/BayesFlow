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

from bayesflow.coupling_networks import AffineCoupling, SplineCoupling
from bayesflow.helper_networks import ActNorm, Orthogonal, Permutation
from bayesflow.inference_networks import InvertibleNetwork


@pytest.mark.parametrize("input_shape", ["2d", "3d"])
@pytest.mark.parametrize("use_soft_flow", [True, False])
@pytest.mark.parametrize("permutation", ["learnable", "fixed"])
@pytest.mark.parametrize("coupling_design", ["affine", "spline", "interleaved"])
@pytest.mark.parametrize("num_coupling_layers", [2, 7])
def test_invertible_network(input_shape, use_soft_flow, permutation, coupling_design, num_coupling_layers):
    """Tests the ``InvertibleNetwork`` core class using a couple of relevant configurations."""

    # Randomize units and input dim
    units = np.random.randint(low=2, high=32)
    input_dim = np.random.randint(low=2, high=32)

    # Create settings dictionaries
    if coupling_design in ["affine", "spline"]:
        coupling_settings = {
            "dense_args": dict(units=units, activation="elu"),
            "num_dense": 1,
        }
    else:
        coupling_settings = {
            "affine": dict(dense_args={"units": units, "activation": "selu"}, num_dense=1),
            "spline": dict(dense_args={"units": units, "activation": "relu"}, bins=8, num_dense=1),
        }

    # Create invertible network with test settings
    network = InvertibleNetwork(
        num_params=input_dim,
        num_coupling_layers=num_coupling_layers,
        use_soft_flow=use_soft_flow,
        permutation=permutation,
        coupling_design=coupling_design,
        coupling_settings=coupling_settings,
    )

    # Create randomized input and output conditions
    batch_size = np.random.randint(low=1, high=32)
    if input_shape == "2d":
        inp = np.random.normal(size=(batch_size, input_dim)).astype(np.float32)
    else:
        n_obs = np.random.randint(low=1, high=32)
        inp = np.random.normal(size=(batch_size, n_obs, input_dim)).astype(np.float32)
    condition_dim = np.random.randint(low=1, high=32)
    condition = np.random.normal(size=(batch_size, condition_dim)).astype(np.float32)

    # Forward and inverse pass
    z, ldj = network(inp, condition)
    z = z.numpy()
    inp_rec = network(z, condition, inverse=True).numpy()

    # Test attributes
    assert network.latent_dim == input_dim
    assert len(network.coupling_layers) == num_coupling_layers
    # Test layer attributes
    for idx, l in enumerate(network.coupling_layers):
        # Permutation
        if permutation == "fixed":
            assert isinstance(l.permutation, Permutation)
        elif permutation == "learnable":
            assert isinstance(l.permutation, Orthogonal)
        # Default ActNorm
        assert isinstance(l.act_norm, ActNorm)
        # Coupling type
        if coupling_design == "affine":
            assert isinstance(l.net1, AffineCoupling) and isinstance(l.net2, AffineCoupling)
        elif coupling_design == "spline":
            assert isinstance(l.net1, SplineCoupling) and isinstance(l.net2, SplineCoupling)
        elif coupling_design == "interleaved":
            if idx % 2 == 0:
                assert isinstance(l.net1, AffineCoupling) and isinstance(l.net2, AffineCoupling)
            else:
                assert isinstance(l.net1, SplineCoupling) and isinstance(l.net2, SplineCoupling)

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
    if input_shape == "2d":
        assert ldj.shape[0] == inp.shape[0]
    else:
        assert ldj.shape[0] == inp.shape[0] and ldj.shape[1] == inp.shape[1]
