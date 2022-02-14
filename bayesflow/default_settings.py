# Copyright 2022 The BayesFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod


class Setting(ABC):
    """Abstract Base class for settings. It's here to potentially extend the setting functionality in future.
    """
    @abstractmethod
    def __init__(self):
        """"""
        pass


class MetaDictSetting(Setting):
    """Implements an interface for a default meta_dict with optional mandatory fields
    """
    def __init__(self, meta_dict: dict, mandatory_fields: list = []):
        """

        Parameters
        ----------
        meta_dict: dict
            Default dictionary.

        mandatory_fields: list, default: []
            List of keys in `meta_dict` that need to be provided by the user.
        """
        
        self.meta_dict = meta_dict
        self.mandatory_fields = mandatory_fields


DEFAULT_SETTING_INVARIANT_NET = MetaDictSetting(
    meta_dict={
        'n_dense_s1': 2,
        'n_dense_s2': 2,
        'n_dense_s3': 2,
        'n_equiv':    2,
        'dense_s1_args': {'activation': 'relu', 'units': 32},
        'dense_s2_args': {'activation': 'relu', 'units': 64},
        'dense_s3_args': {'activation': 'relu', 'units': 32}
    },
    mandatory_fields=[]
)

DEFAULT_SETTING_DENSE_COUPLING = {
    't_args': {
        'dense_args': dict(units=64, kernel_initializer='lecun_normal', activation='selu'),
        'n_dense': 2
    },
    's_args': {
        'dense_args': dict(units=64, kernel_initializer='lecun_normal', activation='selu'),
        'n_dense': 2
    },
}

DEFAULT_SETTING_ATTENTIVE_COUPLING = {
    't_args': {
        'pre_dense_args': dict(units=32, kernel_initializer='lecun_normal', activation='selu'),
        'n_dense_pre': 2,
        'attention_args': dict(key_dim=32, num_heads=4),
        'post_dense_args': dict(units=32, kernel_initializer='lecun_normal', activation='selu'),
        'n_dense_post': 2
    },
    's_args': {
        'pre_dense_args': dict(units=32, kernel_initializer='lecun_normal', activation='selu'),
        'n_dense_pre': 2,
        'attention_args': dict(key_dim=32, num_heads=4),
        'post_dense_args': dict(units=32, kernel_initializer='lecun_normal', activation='selu'),
        'n_dense_post': 2
    }
}

DEFAULT_SETTING_INVERTIBLE_NET = MetaDictSetting(
    meta_dict={
        'n_coupling_layers': 4,
        'coupling_settings': None,
        'coupling_design': 'dense',
        'alpha': 1.9,
        'use_permutation': True,
        'use_act_norm': True,
        'act_norm_init': None,
        'tail_network': None
    },
    mandatory_fields=["n_params"]
)

DEFAULT_SETTING_EVIDENTIAL_NET = MetaDictSetting(
    meta_dict={
        'dense_args': dict(units=128, kernel_initializer='lecun_normal', activation='selu'),
        'n_dense': 3,
        'output_activation': 'softplus'
    },
    mandatory_fields=["n_models"]
)


DEFAULT_SETTING_TAIL_NET = MetaDictSetting(
    meta_dict={
        'dense_args': dict(units=128, kernel_initializer='lecun_normal', activation='selu'),
        'n_dense': 3
    },
    mandatory_fields=[]
)

STRING_CONFIGS = ['var_obs', 'one_hot', 'var_obs_one_hot', 'one_hot_var_obs']


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
    'true_model_indices'            : 'true_model_indices',
    'summary_conditions'            : 'summary_conditions',
    'direct_conditions'             : 'direct_conditions',
    'parameters'                    : 'parameters',
    'observables'                   : 'observables',
    'conditions'                    : 'conditions',
    'posterior_inputs'              : 'posterior_inputs',
    'likelihood_inputs'             : 'likelihood_inputs'
}


MMD_BANDWIDTH_LIST = [
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
    1e3, 1e4, 1e5, 1e6
]