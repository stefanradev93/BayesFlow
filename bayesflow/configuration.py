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

from re import M
from turtle import forward
import numpy as np
from copy import deepcopy

from tensorflow.keras.utils import to_categorical

from bayesflow.default_settings import DEFAULT_KEYS
from bayesflow.exceptions import ConfigurationError


class DefaultJointConfigurator:
    """ Utility class for a generic configrator for joint posterior and likelihood learning.
    """

    def __init__(self, transform_fun=None, combine_fun=None, default_float_type=np.float32):
        
        self.transformer = DefaultJointTransformer() if transform_fun is None else transform_fun
        self.combiner= DefaultJointCombiner() if combine_fun is None else combine_fun
        self.default_float_type = default_float_type

    def __call__(self, forward_dict):
        """ Configures the output of a generative model for joint learning.
        """

        # Default transformer and input
        forward_dict = self.transformer(forward_dict)
        input_dict = self.combiner(forward_dict)

        # Determine and fix float types, if necessary

        input_dict['posterior_inputs'] = {k : v.astype(self.default_float_type) if v is not None else v
                        for k, v in input_dict[DEFAULT_KEYS['posterior_inputs']].items() }
        input_dict['likelihood_inputs'] = {k : v.astype(self.default_float_type) if v is not None else v
                        for k, v in input_dict[DEFAULT_KEYS['likelihood_inputs']].items() }
        return input_dict


class DefaultLikelihoodConfigurator:
    """ Utility class for a generic configrator for likelihood emulation.
    """

    def __init__(self, transform_fun=None, combine_fun=None, default_float_type=np.float32):

        self.transformer = DefaultLikelihoodTransformer() if transform_fun is None else transform_fun
        self.combiner = DefaultLikelihoodCombiner() if combine_fun is None else combine_fun
        self.default_float_type = default_float_type

    def __call__(self, forward_dict):
        """ Configures the output of a generative model for likelihood estimation.
        """

        # Default transformer and input
        forward_dict = self.transformer(forward_dict)
        input_dict = self.combiner(forward_dict)

        # Convert everything to default type or fail gently
        input_dict = {k : v.astype(self.default_float_type) if v is not None else v 
                      for k, v in input_dict.items()}
        return input_dict


class DefaultPosteriorConfigurator:
    """ Utility class for a generic configrator for amortized posterior inference.
    """

    def __init__(self, transform_fun=None, combine_fun=None, default_float_type=np.float32):

        self.transformer = DefaultPosteriorTransformer() if transform_fun is None else transform_fun
        self.combiner = DefaultPosteriorCombiner() if combine_fun is None else combine_fun
        self.default_float_type = default_float_type

    def __call__(self, forward_dict):
        """ Processes the forward dict to configure the input to an amortizer.
        """

        # Default transformer and input
        forward_dict = self.transformer(forward_dict)
        input_dict = self.combiner(forward_dict)

        # Convert everything to default type or fail gently
        input_dict = {k : v.astype(self.default_float_type) if v is not None else v for k, v in input_dict.items()}
        return input_dict


class DefaultModelComparisonConfigurator:
    """ Utility class for a default configurator for amortized model comparison."""

    def __init__(self, n_models, config=None, default_float_type=np.float32):
        
        self.n_models = n_models
        if config is None:
            self.config = DefaultPosteriorConfigurator()
        else:
            self.config = config
        self.default_float_type = default_float_type
        
    def __call__(self, forward_dict):
        """ Convert all variables to arrays and combines them for inference into a dictionary with 
        the following keys, if DEFAULT_KEYS dictionary unchanged: 

        `model_indices`      - the latent model parameters over which a condition density is learned
        `summary_conditions` - the conditioning variables that are first passed through a summary network
        `direct_conditions`  - the conditioning variables that the directly passed to the inference network
        """

        # Prepare placeholders
        out_dict = {
            DEFAULT_KEYS['summary_conditions']: None,
            DEFAULT_KEYS['direct_conditions']: None,
            DEFAULT_KEYS['model_indices']: None
        }
        summary_conditions = []
        direct_conditions = []
        model_indices = []

        # Loop through outputs of individual models
        for m_idx, dict_m in zip(forward_dict[DEFAULT_KEYS['model_indices']], 
                                 forward_dict[DEFAULT_KEYS['model_outputs']]):
            # Configure individual model outputs
            conf_out = self.config(dict_m)
            # Extract summary conditions
            if conf_out.get(DEFAULT_KEYS['summary_conditions']) is not None:
                summary_conditions.append(conf_out[DEFAULT_KEYS['summary_conditions']])
            # Extract direct conditions
            if conf_out.get(DEFAULT_KEYS['direct_conditions']) is not None:
                direct_conditions.append(conf_out[DEFAULT_KEYS['direct_conditions']])
            # Extract model indices as one-hot
            n_draws = dict_m[DEFAULT_KEYS['prior_draws']].shape[0]
            model_indices.append( to_categorical([m_idx] * n_draws, self.n_models) )
        
        # At this point, all elements of the input_dicts should be arrays with identical keys
        out_dict[DEFAULT_KEYS['summary_conditions']] = np.concatenate(summary_conditions) if summary_conditions else None
        out_dict[DEFAULT_KEYS['direct_conditions']] = np.concatenate(direct_conditions) if direct_conditions else None
        out_dict[DEFAULT_KEYS['model_indices']] = np.concatenate(model_indices).astype(self.default_float_type)

        return out_dict


class TransformerUnion:
    """ Utility class for combining the workings of multiple transformers
    """
    def __init__(self, transformers: list):
        self.transformers_list = transformers
    
    def __call__(self, forward_dict):
        """ Applies all transformers to the outputs of a generative model.
        """

        forward_dict = self.transformers_list[0](forward_dict, copy=True)
        for t in self.transformers_list[1:]:
            forward_dict = t(forward_dict, copy=False)
        return forward_dict


class VariableObservationsTransformer:
    """ Utility class for the common case of a simulator with variable observations. 
    Assumes batchable context for simulator is n_obs.
    """

    def __init__(self, default_transform=np.log1p):

        self.n_obs_to_array = lambda N, B: default_transform(N * np.ones((B, 1)))

    def __call__(self, forward_dict, copy=True):
        """ Transform integer n_obs to an array of size (batch_size, 1), which can be
        concatenated along the (summarized data) and passed as a direct condition for the
        invertible inference network.
        """

        if copy:
            forward_dict = deepcopy(forward_dict)

        # Convert n_obs to vector and transform to sqrt
        N = forward_dict[DEFAULT_KEYS['sim_non_batchable_context']]
        B = forward_dict[DEFAULT_KEYS['prior_draws']].shape[0]
        forward_dict[DEFAULT_KEYS['sim_non_batchable_context']] = self.n_obs_to_array(N, B)
        return forward_dict


class OneHotTransformer:
    """ Utility class for the common case of integer batchable context.
    """

    def __init__(self, n_categories_prior_context=None, n_categories_sim_context=None):
        """ Creates an instance of a utility one-hot-transofmration class. The user is advised to
        specify the number of categories for prior and simulator context, otherwise erroneous cases
        might arise during batch simulations if a batch contains (as per randomness) less categories
        than actually present.
        """

        self.n_categories_prior_context = n_categories_prior_context
        self.n_categories_sim_context = n_categories_sim_context

    def __call__(self, forward_dict, copy=True):
        """ Transform integer n_obs to an array of size (batch_size, 1), which can be
        concatenated along the (summarized data) and passed as a condition for the
        invertible network.
        """

        if copy:
            forward_dict = deepcopy(forward_dict)
        
        # Transform prior batchable context
        if forward_dict[DEFAULT_KEYS['prior_batchable_context']] is not None:
            forward_dict[DEFAULT_KEYS['prior_batchable_context']] = to_categorical(
                    forward_dict[DEFAULT_KEYS['prior_batchable_context']],
                    self.n_categories_prior_context)

        # Transform simulator batchable context
        if forward_dict[DEFAULT_KEYS['sim_batchable_context']] is not None:
            forward_dict[DEFAULT_KEYS['sim_batchable_context']] = to_categorical(
                    forward_dict[DEFAULT_KEYS['sim_batchable_context']],
                    self.n_categories_sim_context)
        return forward_dict


class DefaultPosteriorCombiner:
    """ Default combiner attempts to convert all variables to a BayesFlow-compatible format. 
    Assumes all existing batchable context to be concatenated with the data and subsequently passed
    through a summary network and all existing non-batchable context to be concatenated with the
    outputs of the summary network.
    """
    
    def __call__(self, forward_dict):
        """ Convert all variables to arrays and combines them for inference into a dictionary with 
        the following keys, if DEFAULT_KEYS dictionary unchanged: 

        `parameters`         - the latent model parameters over which a condition density is learned
        `summary_conditions` - the conditioning variables that are first passed through a summary network
        `direct_conditions`  - the conditioning variables that the directly passed to the inference network

        Parameters
        ----------
        forward_dict : dict  
            Input dictionary containing the following mandatory keys, if DEFAULT_KEYS dictionary unchanged: 
            # TODO
        """
        
        # Prepare placeholder
        out_dict = {
            DEFAULT_KEYS['parameters']: None,
            DEFAULT_KEYS['summary_conditions']: None,
            DEFAULT_KEYS['direct_conditions']: None
        }

        # Pushforward target are the parameters
        out_dict[DEFAULT_KEYS['parameters']] = forward_dict[DEFAULT_KEYS['prior_draws']]


        # Determine whether simulated or observed data available, throw if None present
        if forward_dict.get(DEFAULT_KEYS['sim_data']) is None and \
           forward_dict.get(DEFAULT_KEYS['obs_data']) is None:

           raise ConfigurationError(f"Either {DEFAULT_KEYS['sim_data']} or {DEFAULT_KEYS['obs_data']}" + 
                                    " should be present as keys in the forward_dict, but not both!")

        # If only simulated or observed data present, all good
        elif forward_dict.get(DEFAULT_KEYS['sim_data']) is not None:
            data = forward_dict.get(DEFAULT_KEYS['sim_data'])
        elif forward_dict.get(DEFAULT_KEYS['obs_data']) is not None:
            data = forward_dict.get(DEFAULT_KEYS['obs_data'])
        
        # Else if neither 'sim_data' nor 'obs_data' present, throw again
        else:
            raise ConfigurationError(f"Either {DEFAULT_KEYS['sim_data']} or {DEFAULT_KEYS['obs_data']}" + 
                                    " should be present as keys in the forward_dict.")

        # Handle simulated or observed data or throw if the data could not be converted to an array 
        try:
            if type(data) is not np.ndarray:
                summary_conditions = np.array(data)
            else:
                summary_conditions = data
        except Exception as _:
            raise ConfigurationError("Could not convert input data to array...")
        
        # Handle prior batchable context or throw if error encountered
        if forward_dict.get(DEFAULT_KEYS['prior_batchable_context']) is not None:
            try:
                if type(forward_dict[DEFAULT_KEYS['prior_batchable_context']]) is not np.ndarray:
                    pbc_as_array = np.array(forward_dict[DEFAULT_KEYS['prior_batchable_context']])
                else:
                    pbc_as_array = forward_dict[DEFAULT_KEYS['prior_batchable_context']]
            except Exception as _:
                raise ConfigurationError("Could not convert prior batchable context to array.")
                
            try:
                summary_conditions = np.concatenate([summary_conditions, pbc_as_array], axis=-1)
            except Exception as _:
                raise ConfigurationError(f"Could not concatenate data and prior batchable context. Shape mismatch: " +
                                          "data - {summary_conditions.shape}, prior_batchable_context - {pbc_as_array.shape}.")

        # Handle simulation batchable context, or throw if error encountered
        if forward_dict.get(DEFAULT_KEYS['sim_batchable_context']) is not None:
            try:
                if type(forward_dict[DEFAULT_KEYS['sim_batchable_context']]) is not np.ndarray:
                    sbc_as_array = np.array(forward_dict[DEFAULT_KEYS['sim_batchable_context']])
                else:
                    sbc_as_array = forward_dict[DEFAULT_KEYS['sim_batchable_context']]
            except Exception as _:
                raise ConfigurationError("Could not convert simulation batchable context to array!")
                
            try:
                summary_conditions = np.concatenate([summary_conditions, sbc_as_array], axis=-1)
            except Exception as _:
                raise ConfigurationError(f"Could not concatenate data (+optional prior context) and" +
                                         f" simulation batchable context. Shape mismatch:" + 
                                         f" data - {summary_conditions.shape}, prior_batchable_context - {sbc_as_array.shape}")
        
        # Add summary conditions to output dict
        out_dict[DEFAULT_KEYS['summary_conditions']] = summary_conditions

        # Handle non-batchable contexts
        if forward_dict.get(DEFAULT_KEYS['prior_non_batchable_context']) is None and \
           forward_dict.get(DEFAULT_KEYS['sim_non_batchable_context']) is None:
            return out_dict

        # Handle prior non-batchable context
        direct_conditions = None
        if forward_dict.get(DEFAULT_KEYS['prior_non_batchable_context']) is not None:
            try:
                if type(forward_dict[DEFAULT_KEYS['prior_non_batchable_context']]) is not np.ndarray:
                    pnbc_conditions = np.array(forward_dict[DEFAULT_KEYS['prior_non_batchable_context']])
                else:
                    pnbc_conditions = forward_dict[DEFAULT_KEYS['prior_non_batchable_context']]
            except Exception as _:
                raise ConfigurationError("Could not convert prior non_batchable_context to an array!")
            direct_conditions = pnbc_conditions

        # Handle simulation non-batchable context
        if forward_dict.get(DEFAULT_KEYS['sim_non_batchable_context']) is not None:
            try:
                if type(forward_dict[DEFAULT_KEYS['sim_non_batchable_context']]) is not np.ndarray:
                    snbc_conditions = np.array(forward_dict[DEFAULT_KEYS['sim_non_batchable_context']])
                else:
                    snbc_conditions = forward_dict[DEFAULT_KEYS['sim_non_batchable_context']]
            except Exception as _:
                raise ConfigurationError("Could not convert sim_non_batchable_context to array!")
            try:
                if direct_conditions is not None:
                    direct_conditions = np.concatenate([direct_conditions, snbc_conditions], axis=-1)
                else:
                    direct_conditions = snbc_conditions
            except Exception as _:
                raise ConfigurationError(f"Could not concatenate prior non-batchable context and  \
                            simulation non-batchable context. Shape mismatch: \
                                - {direct_conditions.shape} vs. {snbc_conditions.shape}")
        out_dict[DEFAULT_KEYS['direct_conditions']] = direct_conditions
        
        return out_dict


class DefaultLikelihoodCombiner:
    def __call__(self, forward_dict):

        # Prepare placeholder
        out_dict = {
            DEFAULT_KEYS['observables']: None,
            DEFAULT_KEYS['conditions']: None
        }

        # Determine whether simulated or observed data available, throw if None present
        if forward_dict.get(DEFAULT_KEYS['sim_data']) is None and \
           forward_dict.get(DEFAULT_KEYS['obs_data']) is None:

           raise ConfigurationError(f"Either {DEFAULT_KEYS['sim_data']} or {DEFAULT_KEYS['obs_data']}" + 
                                    " should be present as keys in the forward_dict.")

        # If only simulated or observed data present, all good
        elif forward_dict.get(DEFAULT_KEYS['sim_data']) is not None:
            data = forward_dict.get(DEFAULT_KEYS['sim_data'])
        elif forward_dict.get(DEFAULT_KEYS['obs_data']) is not None:
            data = forward_dict.get(DEFAULT_KEYS['obs_data'])
        
        # Else if neither 'sim_data' nor 'obs_data' present, throw again
        else:
            raise ConfigurationError(f"Either {DEFAULT_KEYS['sim_data']} or {DEFAULT_KEYS['obs_data']}" + 
                                    " should be present as keys in the forward_dict.")

        # Extract targets and conditions
        out_dict[DEFAULT_KEYS['observables']] = data
        out_dict[DEFAULT_KEYS['conditions']] = forward_dict[DEFAULT_KEYS['prior_draws']]

        return out_dict


class DefaultJointCombiner:
    def __init__(self, posterior_combiner=None, likelihood_combiner=None):

        if posterior_combiner is None:
            self.posterior_combiner = DefaultPosteriorCombiner()
        else:
            self.posterior_combiner = posterior_combiner

        if likelihood_combiner is None:
            self.likelihood_combiner = DefaultLikelihoodCombiner()
        else:
            self.likelihood_combiner = likelihood_combiner

    def __call__(self, forward_dict):

        # Prepare placeholder for output dictionary
        out_dict = {
            DEFAULT_KEYS['likelihood_inputs']: None,
            DEFAULT_KEYS['posterior_inputs']: None
        }

        # Populate output dictionary
        out_dict[DEFAULT_KEYS['posterior_inputs']] = self.posterior_combiner(forward_dict)
        out_dict[DEFAULT_KEYS['likelihood_inputs']] = self.likelihood_combiner(forward_dict)

        return out_dict


class DefaultJointTransformer:
    """TODO"""

    def __call__(self, forward_dict):
        return forward_dict


class DefaultPosteriorTransformer:
    """TODO"""
    def __call__(self, forward_dict):
        return forward_dict


class DefaultLikelihoodTransformer:
    """TODO"""
    def __call__(self, forward_dict):
        return forward_dict


