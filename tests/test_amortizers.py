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

from bayesflow.amortizers import AmortizedLikelihood, AmortizedPosterior, AmortizedPosteriorLikelihood
from bayesflow.default_settings import DEFAULT_KEYS
from bayesflow.networks import InvariantNetwork, InvertibleNetwork


@pytest.mark.parametrize("cond_shape", ["2d", "3d"])
@pytest.mark.parametrize("summary_loss", ["MMD", None])
@pytest.mark.parametrize("soft", [True, False])
def test_amortized_posterior(cond_shape, summary_loss, soft):
    """Tests the ``AmortizedPosterior`` instance with relevant configurations."""

    # Randomize input
    batch_size = np.random.randint(low=1, high=32)
    inp_dim = np.random.randint(low=2, high=32)
    cond_dim = np.random.randint(low=2, high=32)

    # Create settings dictionaries for inference network
    dense_net_settings = {
        "dense_args": dict(units=8, activation="elu"),
        "num_dense": 1,
        "spec_norm": True,
    }

    # Create inference network instance
    inference_network = InvertibleNetwork(
        **{"num_params": inp_dim, "use_soft_flow": soft, "coupling_settings": dense_net_settings}
    )

    # Create summary network and condition
    if cond_shape == "3d":
        n_obs = np.random.randint(low=1, high=32)
        condition = np.random.normal(size=(batch_size, n_obs, cond_dim)).astype(np.float32)
        summary_network = InvariantNetwork()
    else:
        condition = np.random.normal(size=(batch_size, cond_dim)).astype(np.float32)
        summary_network = None
        summary_loss = None

    target = np.random.normal(size=(batch_size, inp_dim)).astype(np.float32)

    # Create amortizer instance
    amortizer = AmortizedPosterior(inference_network, summary_network, summary_loss_fun=summary_loss)

    # Prepare input
    if cond_shape == "3d":
        inp_dict = {DEFAULT_KEYS["parameters"]: target, DEFAULT_KEYS["summary_conditions"]: condition}
    else:
        inp_dict = {DEFAULT_KEYS["parameters"]: target, DEFAULT_KEYS["direct_conditions"]: condition}

    # Pass through network
    out = amortizer(inp_dict)
    z, ldj = out

    # Compute loss
    loss = amortizer.compute_loss(inp_dict).numpy()

    # Compute lpdf
    log_post = amortizer.log_posterior(inp_dict)
    lpdf = amortizer.log_prob(inp_dict)

    # Sampling
    n_samples = np.random.randint(low=1, high=200)
    samples = amortizer.sample(inp_dict, n_samples)

    # Test output types and shapes
    assert type(out) is tuple
    assert z.shape[0] == batch_size
    assert z.shape[1] == inp_dim
    assert ldj.shape[0] == batch_size

    # Test attributes
    assert amortizer.latent_is_dynamic is False
    assert amortizer.latent_dim == inp_dim
    if cond_shape == "3d":
        assert amortizer.summary_net is not None
    else:
        assert amortizer.summary_net is None

    # Test loss is a single float
    assert type(loss) is np.float32

    # Test log posterior and lpdf shapes
    assert log_post.shape[0] == batch_size
    assert lpdf.shape[0] == batch_size

    # Log posterior and lpdf should be the same, unless using softflow
    # which will introduce some noise in the untrained version
    if not soft:
        assert np.allclose(log_post, lpdf, atol=1e-5)

    # Test shape of samples
    if batch_size == 1:
        assert samples.shape[0] == n_samples
        assert samples.shape[1] == inp_dim
    else:
        assert samples.shape[0] == batch_size
        assert samples.shape[1] == n_samples
        assert samples.shape[2] == inp_dim


@pytest.mark.parametrize("inp_shape", ["2d", "3d"])
@pytest.mark.parametrize("soft", [True, False])
def test_amortized_likelihood(inp_shape, soft):
    """Tests the ``AmortizedLikelihood`` instance with relevant configurations."""

    # Randomize input
    batch_size = np.random.randint(low=1, high=32)
    inp_dim = np.random.randint(low=2, high=32)
    cond_dim = np.random.randint(low=2, high=32)
    units = np.random.randint(low=2, high=32)

    # Create settings dictionaries for inference network
    dense_net_settings = {
        "dense_args": dict(units=units, kernel_initializer="glorot_uniform", activation="elu"),
        "num_dense": 1,
        "spec_norm": False,
    }

    # Create inference network instance
    surrogate_network = InvertibleNetwork(
        **{"num_params": inp_dim, "use_soft_flow": soft, "coupling_settings": dense_net_settings}
    )

    # Create input and condition
    if inp_shape == "3d":
        n_obs = np.random.randint(low=1, high=32)
        inp = np.random.normal(size=(batch_size, n_obs, inp_dim)).astype(np.float32)
    else:
        inp = np.random.normal(size=(batch_size, inp_dim)).astype(np.float32)
    condition = np.random.normal(size=(batch_size, cond_dim)).astype(np.float32)

    # Create amortizer instance
    amortizer = AmortizedLikelihood(surrogate_network)

    # Create input dictionary
    inp_dict = {DEFAULT_KEYS["observables"]: inp, DEFAULT_KEYS["conditions"]: condition}

    # Pass through network
    out = amortizer(inp_dict)
    z, ldj = out

    # Compute loss
    loss = amortizer.compute_loss(inp_dict).numpy()

    # Compute lpdf
    log_lik = amortizer.log_likelihood(inp_dict)
    lpdf = amortizer.log_prob(inp_dict)

    # Sampling
    n_samples = np.random.randint(low=1, high=200)
    samples = amortizer.sample(inp_dict, n_samples)

    # Test output types and shapes
    assert type(out) is tuple
    assert z.shape[0] == batch_size
    assert ldj.shape[0] == batch_size

    if inp_shape == "3d":
        assert z.shape[1] == n_obs
        assert z.shape[2] == inp_dim
        assert ldj.shape[1] == n_obs
    else:
        assert z.shape[1] == inp_dim

    # Test attributes
    assert amortizer.latent_dim == inp_dim

    # Test loss is a single float
    assert type(loss) is np.float32

    # Test log posterior and lpdf shapes
    assert log_lik.shape[0] == batch_size
    assert lpdf.shape[0] == batch_size
    if inp_shape == "3d":
        assert log_lik.shape[1] == n_obs
        assert lpdf.shape[1] == n_obs

    # Log posterior and lpdf should be the same, unless using softflow
    # which will introduce some noise in the untrained version
    if not soft:
        assert np.allclose(log_lik, lpdf, atol=1e-5)

    # Test shape of samples
    if batch_size == 1:
        assert samples.shape[0] == n_samples
        assert samples.shape[1] == inp_dim
    else:
        assert samples.shape[0] == batch_size
        assert samples.shape[1] == n_samples
        assert samples.shape[2] == inp_dim


@pytest.mark.parametrize("data_dim", [12, 3])
@pytest.mark.parametrize("params_dim", [4, 8])
def test_joint_amortizer(data_dim, params_dim):
    """Tests the ``JointAmortizer`` instance with relevant configurations."""

    # Randomize input
    batch_size = np.random.randint(low=1, high=32)
    units = np.random.randint(low=2, high=32)

    # Create settings dictionaries for inference network
    dense_net_settings = {
        "dense_args": dict(units=units, activation="elu"),
        "num_dense": 1,
        "spec_norm": True,
    }

    # Create amortizers
    p_amortizer = AmortizedPosterior(
        InvertibleNetwork(**{"num_params": params_dim, "coupling_settings": dense_net_settings})
    )
    l_amortizer = AmortizedLikelihood(
        InvertibleNetwork(**{"num_params": data_dim, "coupling_settings": dense_net_settings})
    )
    amortizer = AmortizedPosteriorLikelihood(p_amortizer, l_amortizer)

    # Create inputs and format into a dictionary
    params = np.random.normal(size=(batch_size, params_dim)).astype(np.float32)
    data = np.random.normal(size=(batch_size, data_dim)).astype(np.float32)
    inp_dict = {}
    inp_dict[DEFAULT_KEYS["posterior_inputs"]] = {
        DEFAULT_KEYS["parameters"]: params,
        DEFAULT_KEYS["direct_conditions"]: data,
    }
    inp_dict[DEFAULT_KEYS["likelihood_inputs"]] = {
        DEFAULT_KEYS["observables"]: data,
        DEFAULT_KEYS["conditions"]: params,
    }

    # Compute lpdf
    log_lik = amortizer.log_likelihood(inp_dict)
    log_post = amortizer.log_posterior(inp_dict)

    # Sampling
    n_samples_p = np.random.randint(low=1, high=200)
    n_samples_l = np.random.randint(low=1, high=200)
    p_samples = amortizer.sample_parameters(inp_dict, n_samples_p)
    l_samples = amortizer.sample_data(inp_dict, n_samples_l)

    # Check shapes
    assert log_lik.shape[0] == batch_size
    assert log_post.shape[0] == batch_size
    if batch_size == 1:
        assert p_samples.shape[0] == n_samples_p
        assert l_samples.shape[0] == n_samples_l
    else:
        assert p_samples.shape[0] == batch_size
        assert l_samples.shape[0] == batch_size
        assert p_samples.shape[1] == n_samples_p
        assert l_samples.shape[1] == n_samples_l
        assert p_samples.shape[2] == params_dim
        assert l_samples.shape[2] == data_dim
