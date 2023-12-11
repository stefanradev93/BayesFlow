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

from abc import ABC, abstractmethod

import tensorflow as tf


class Setting(ABC):
    """Abstract base class for settings. It's here to potentially extend the setting functionality in future."""

    @abstractmethod
    def __init__(self):
        """"""
        pass


class MetaDictSetting(Setting):
    """Implements an interface for a default meta_dict with optional mandatory fields."""

    def __init__(self, meta_dict: dict, mandatory_fields: list = []):
        """

        Parameters
        ----------
        meta_dict        : dict
            Default dictionary.
        mandatory_fields : list, default: []
            List of keys in `meta_dict` that need to be provided by the user.
        """

        self.meta_dict = meta_dict
        self.mandatory_fields = mandatory_fields


DEFAULT_SETTING_INVARIANT_NET = MetaDictSetting(
    meta_dict={
        "num_dense_s1": 2,
        "num_dense_s2": 2,
        "num_dense_s3": 2,
        "num_equiv": 2,
        "pooling_fun": "mean",
        "dense_s1_args": None,
        "dense_s2_args": None,
        "dense_s3_args": None,
        "summary_dim": 10,
    },
    mandatory_fields=[],
)


DEFAULT_SETTING_MULTI_CONV = {
    "layer_args": {"activation": "relu", "filters": 32, "strides": 1, "padding": "causal"},
    "min_kernel_size": 1,
    "max_kernel_size": 3,
}


DEFAULT_SETTING_DENSE_DEEP_SET = {"units": 64, "activation": "relu", "kernel_initializer": "glorot_uniform"}


DEFAULT_SETTING_DENSE_RECT = {"units": 256, "activation": "swish", "kernel_initializer": "glorot_uniform"}


DEFAULT_SETTING_DENSE_ATTENTION = {"units": 64, "activation": "relu", "kernel_initializer": "glorot_uniform"}


DEFAULT_SETTING_DENSE_EVIDENTIAL = {
    "units": 64,
    "kernel_initializer": "glorot_uniform",
    "activation": "elu",
}


DEFAULT_SETTING_DENSE_PMP = {
    "units": 64,
    "kernel_initializer": "glorot_uniform",
    "activation": "elu",
}


DEFAULT_SETTING_AFFINE_COUPLING = MetaDictSetting(
    meta_dict={
        "dense_args": dict(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        "num_dense": 2,
        "spec_norm": False,
        "mc_dropout": False,
        "dropout": True,
        "residual": False,
        "dropout_prob": 0.05,
        "soft_clamping": 1.9,
    },
    mandatory_fields=[],
)


DEFAULT_SETTING_SPLINE_COUPLING = MetaDictSetting(
    meta_dict={
        "dense_args": dict(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        "num_dense": 2,
        "spec_norm": False,
        "mc_dropout": False,
        "dropout": True,
        "residual": False,
        "dropout_prob": 0.05,
        "bins": 16,
        "default_domain": (-5.0, 5.0, -5.0, 5.0),
    },
    mandatory_fields=[],
)


DEFAULT_SETTING_ATTENTION = {"key_dim": 32, "num_heads": 4, "dropout": 0.01}


DEFAULT_SETTING_INVERTIBLE_NET = MetaDictSetting(
    meta_dict={
        "num_coupling_layers": 5,
        "coupling_net_settings": None,
        "coupling_design": "affine",
        "permutation": "fixed",
        "use_act_norm": True,
        "act_norm_init": None,
        "use_soft_flow": False,
        "soft_flow_bounds": (1e-3, 5e-2),
    },
    mandatory_fields=["num_params"],
)


DEFAULT_SETTING_EVIDENTIAL_NET = MetaDictSetting(
    meta_dict={
        "dense_args": dict(units=128, activation="relu"),
        "num_dense": 3,
        "output_activation": "softplus",
    },
    mandatory_fields=["num_models"],
)


DEFAULT_SETTING_PMP_NET = MetaDictSetting(
    meta_dict={
        "dense_args": dict(units=64, activation="relu"),
        "num_dense": 3,
        "output_activation": "softmax",
    },
    mandatory_fields=["num_models"],
)


OPTIMIZER_DEFAULTS = {"global_clipnorm": 1.0}


DEFAULT_KEYS = {
    "prior_draws": "prior_draws",
    "obs_data": "obs_data",
    "sim_data": "sim_data",
    "batchable_context": "batchable_context",
    "non_batchable_context": "non_batchable_context",
    "prior_batchable_context": "prior_batchable_context",
    "prior_non_batchable_context": "prior_non_batchable_context",
    "prior_context": "prior_context",
    "hyper_prior_draws": "hyper_prior_draws",
    "shared_prior_draws": "shared_prior_draws",
    "local_prior_draws": "local_prior_draws",
    "sim_batchable_context": "sim_batchable_context",
    "sim_non_batchable_context": "sim_non_batchable_context",
    "summary_conditions": "summary_conditions",
    "direct_conditions": "direct_conditions",
    "parameters": "parameters",
    "hyper_parameters": "hyper_parameters",
    "shared_parameters": "shared_parameters",
    "local_parameters": "local_parameters",
    "observables": "observables",
    "targets": "targets",
    "conditions": "conditions",
    "posterior_inputs": "posterior_inputs",
    "likelihood_inputs": "likelihood_inputs",
    "model_outputs": "model_outputs",
    "model_indices": "model_indices",
}


MMD_BANDWIDTH_LIST = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]

# Minimum time interval between tqdm status updates to reduce
# load. Only respected when refresh=False in set_postfix
# and set_postfix_str
TQDM_MININTERVAL = 0.1  # in seconds
