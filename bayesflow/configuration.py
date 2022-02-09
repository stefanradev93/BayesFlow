import numpy as np
from copy import deepcopy

from tensorflow.keras.utils import to_categorical

from bayesflow.exceptions import ConfigurationError


class DefaultJointConfigurator:
    """ Utility class for a generic configrator for joint posterior and likelihood learning.
    """

    def __init__(self, transform_fun=None, combine_fun=None):
        
        self.transformer = DefaultJointTransformer() if transform_fun is None else transform_fun
        self.combiner= DefaultJointCombiner() if combine_fun is None else combine_fun

    def __call__(self, forward_dict):
        """ Configures the output of a generative model for joint learning.
        """

        # Default transformer and input
        forward_dict = self.transformer(forward_dict)
        input_dict = self.combiner(forward_dict)
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
        input_dict = {k : v.astype(self.default_float_type) for k, v in input_dict.items()}
        return input_dict


class DefaultPosteriorConfigurator:
    """ Utility class for a generic configrator for posterior inference.
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
        input_dict = {k : v.astype(self.default_float_type) for k, v in input_dict.items()}
        return input_dict


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
        concatenated along the (summarized data) and passed as a condition for the
        invertible network.
        """

        if copy:
            forward_dict = deepcopy(forward_dict)

        # Convert n_obs to vector and transform to sqrt
        N = forward_dict['sim_non_batchable_context']
        B = forward_dict['prior_draws'].shape[0]
        forward_dict['sim_non_batchable_context'] = self.n_obs_to_array(N, B)
        return forward_dict


class OneHotTransformer:
    """ Utility class for the common case of integer batchable context.
    """

    def __init__(self, n_categories_prior_context=None, n_categories_sim_context=None):

        self.n_categories_prior_context = n_categories_prior_context
        self.n_categories_sim_context = n_categories_sim_context

    def __call__(self, forward_dict, copy=True):
        """ Transform integer n_obs to an array of size (batch_size, 1), which can be
        concatenated along the (summarized data) and passed as a condition for the
        invertible network.
        """

        if copy:
            forward_dict = deepcopy(forward_dict)

        if forward_dict['prior_batchable_context'] is not None:
            forward_dict['prior_batchable_context'] = to_categorical(
                    forward_dict['prior_batchable_context'],
                    self.n_categories_prior_context
            )
        if forward_dict['sim_batchable_context'] is not None:
            forward_dict['sim_batchable_context'] = to_categorical(
                    forward_dict['sim_batchable_context'],
                    self.n_categories_sim_context
            )
        return forward_dict


class DefaultPosteriorCombiner:
    """ Default combiner attempts to convert all variables to a BayesFlow-compatible format. 
    Assumes all existing batchable context to be concatenated with the data and subsequently passed
    through a summary network and all existing non-batchable context to be concatenated with the
    outputs of the summary network.
    """
    
    def __call__(self, forward_dict):
        """ Convert all variables to arrays and combines them for inference into a dictionary with following keys:
        'parameters' - the pushforward quantities for an inference network (e.g., parameters)
        'summary_conditions' - the quantities that will first be passed through a summary network
        'direct_conditions' - the quantities that will be used to condition the inference network directly
        TODO
        """
        
        # Prepare placeholder
        out_dict = {
            'parameters': None,
            'summary_conditions': None,
            'direct_conditions': None
        }

        # Assume prior_draws contains all pushforward quantities
        out_dict['parameters'] = forward_dict['prior_draws']
        
        # Handle data
        try:
            if type(forward_dict['sim_data']) is not np.ndarray:
                summary_conditions = np.array(forward_dict['sim_data'])
            else:
                summary_conditions = forward_dict['sim_data']
        except Exception as _:
            raise ConfigurationError("Could not convert data to array...")
        
        # Handle prior batchable context
        if forward_dict['prior_batchable_context'] is not None:
            try:
                if type(forward_dict['prior_batchable_context']) is not np.ndarray:
                    pbc_as_array = np.array(forward_dict['prior_batchable_context'])
                else:
                    pbc_as_array = forward_dict['prior_batchable_context']
            except Exception as _:
                raise ConfigurationError("Could not convert prior batchable context to array")
                
            try:
                summary_conditions = np.concatenate([summary_conditions, pbc_as_array], axis=-1)
            except Exception as _:
                raise ConfigurationError(f"Could not concatenate data and prior batchable context. Shape mismatch: " +
                                          "data - {summary_conditions.shape}, prior_batchable_context - {pbc_as_array.shape}.")

        # Handle simulation batchable context
        if forward_dict['sim_batchable_context'] is not None:
            try:
                if type(forward_dict['sim_batchable_context']) is not np.ndarray:
                    sbc_as_array = np.array(forward_dict['sim_batchable_context'])
                else:
                    sbc_as_array = forward_dict['sim_batchable_context']
            except Exception as _:
                raise ConfigurationError("Could not convert simulation batchable context to array")
                
            try:
                summary_conditions = np.concatenate([summary_conditions, sbc_as_array], axis=-1)
            except Exception as _:
                raise ConfigurationError(f"Could not concatenate data (+optional prior context) and simulation batchable context. Shape mismatch: \
                            data - {summary_conditions.shape}, prior_batchable_context - {sbc_as_array.shape}")
        out_dict['summary_conditions'] = summary_conditions
        
        # Handle non-batchable contexts
        if forward_dict['prior_non_batchable_context'] is None and forward_dict['sim_non_batchable_context'] is None:
            return out_dict

        # Prior non-batchable context
        conditions = None
        if forward_dict['prior_non_batchable_context'] is not None:
            try:
                if type(forward_dict['prior_non_batchable_context']) is not np.ndarray:
                    pnbc_conditions = np.array(forward_dict['prior_non_batchable_context'])
                else:
                    pnbc_conditions = forward_dict['prior_non_batchable_context']
            except Exception as _:
                raise ConfigurationError("Could not convert prior non_batchable_context to array")
            conditions = pnbc_conditions

        # Simulation non-batchable context
        if forward_dict['sim_non_batchable_context'] is not None:
            try:
                if type(forward_dict['sim_non_batchable_context']) is not np.ndarray:
                    snbc_conditions = np.array(forward_dict['sim_non_batchable_context'])
                else:
                    snbc_conditions = forward_dict['sim_non_batchable_context']
            except Exception as _:
                raise ConfigurationError("Could not convert sim_non_batchable_context to array")
            try:
                if conditions is not None:
                    conditions = np.concatenate([conditions, snbc_conditions], axis=-1)
                else:
                    conditions = snbc_conditions
            except Exception as _:
                raise ConfigurationError(f"Could not concatenate prior non-batchable context and  \
                            simulation non-batchable context. Shape mismatch: \
                                - {conditions.shape} vs. {snbc_conditions.shape}")
        
        out_dict['direct_conditions'] = conditions
        return out_dict


class DefaultLikelihoodCombiner:
    def __call__(self, forward_dict):

        # Prepare placeholder
        out_dict = {
            'data': None,
            'conditions': None
        }

        # Extract targets and conditions
        out_dict['data'] = forward_dict['sim_data']
        out_dict['conditions'] = forward_dict['prior_draws']

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

        # Prepare placeholder
        out_dict = {
            'likelihood': None,
            'posterior': None
        }

        out_dict['posterior'] = self.posterior_combiner(forward_dict)
        out_dict['likelihood'] = self.likelihood_combiner(forward_dict)

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


