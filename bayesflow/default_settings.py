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
        'num_dense_s1'    : 2,
        'num_dense_s2'    : 2,
        'num_dense_s3'    : 2,
        'num_equiv'       : 2,
        'pooling_fun'     : 'mean',
        'dense_s1_args' : None,
        'dense_s2_args' : None,
        'dense_s3_args' : None,
        'summary_dim'   : 10
    },
    mandatory_fields=[]
)


DEFAULT_SETTING_MULTI_CONV = {
    'layer_args': {
        'activation': 'swish',
        'filters': 32,
        'strides': 1,
        'padding': 'causal'
    },
    'min_kernel_size': 1,
    'max_kernel_size': 3
}


DEFAULT_SETTING_DENSE_INVARIANT = {
    'units': 64,
    'activation': 'relu',  
    'kernel_initializer': 'glorot_uniform'
}


DEFAULT_SETTING_DENSE_EVIDENTIAL = {
    'units': 128,
    'kernel_initializer': 'glorot_uniform',
    'activation': 'elu',
}


DEFAULT_SETTING_DENSE_COUPLING = MetaDictSetting(
    meta_dict={
        't_args': {
            'dense_args': dict(units=128, kernel_initializer='glorot_uniform', activation='tanh'),
            'num_dense': 2,
            'spec_norm': False,
            'mc_dropout': False,
            'dropout': False,
            'dropout_prob': 0.1
        },
        's_args': {
            'dense_args': dict(units=128, kernel_initializer='glorot_uniform', activation='tanh'),
            'num_dense': 2,
            'spec_norm': False,
            'mc_dropout': False,
            'dropout': False,
            'dropout_prob': 0.1
        },
    },
    mandatory_fields=[]
)


DEFAULT_SETTING_INVERTIBLE_NET = MetaDictSetting(
    meta_dict={
        'num_coupling_layers': 6,
        'coupling_net_settings': None,
        'coupling_design': 'dense',
        'soft_clamping': 1.9,
        'use_permutation': True,
        'use_act_norm': True,
        'act_norm_init': None,
        'use_soft_flow': False,
        'soft_flow_bounds': (1e-3, 5e-2),
    },
    mandatory_fields=["num_params"]
)


DEFAULT_SETTING_EVIDENTIAL_NET = MetaDictSetting(
    meta_dict={
        'dense_args': dict(units=128, kernel_initializer='glorot_uniform', activation='relu'),
        'num_dense': 3,
        'output_activation': 'softplus'
    },
    mandatory_fields=["num_models"]
)


OPTIMIZER_DEFAULTS = {
    'global_clipnorm': 1.
}


DEFAULT_KEYS = {
    'prior_draws'                   : 'prior_draws',
    'obs_data'                      : 'obs_data',
    'sim_data'                      : 'sim_data',
    'batchable_context'             : 'batchable_context',
    'non_batchable_context'         : 'non_batchable_context',
    'prior_batchable_context'       : 'prior_batchable_context',
    'prior_non_batchable_context'   : 'prior_non_batchable_context',
    'sim_batchable_context'         : 'sim_batchable_context',
    'sim_non_batchable_context'     : 'sim_non_batchable_context',
    'summary_conditions'            : 'summary_conditions',
    'direct_conditions'             : 'direct_conditions',
    'parameters'                    : 'parameters',
    'observables'                   : 'observables',
    'conditions'                    : 'conditions',
    'posterior_inputs'              : 'posterior_inputs',
    'likelihood_inputs'             : 'likelihood_inputs',
    'model_outputs'                 : 'model_outputs',
    'model_indices'                 : 'model_indices'
}


MMD_BANDWIDTH_LIST = [
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
    1e3, 1e4, 1e5, 1e6
]
