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
from tensorflow.keras.utils import to_categorical

from bayesflow.default_settings import DEFAULT_KEYS
from bayesflow.exceptions import ConfigurationError


class DefaultJointConfigurator:
    """Fallback class for a generic configurator for joint posterior and likelihood approximation."""

    def __init__(self, default_float_type=np.float32):
        self.posterior_config = DefaultPosteriorConfigurator(default_float_type=default_float_type)
        self.likelihood_config = DefaultLikelihoodConfigurator(default_float_type=default_float_type)
        self.default_float_type = default_float_type

    def __call__(self, forward_dict):
        """Configures the outputs of a generative model for joint learning."""

        input_dict = {}
        input_dict[DEFAULT_KEYS["posterior_inputs"]] = self.posterior_config(forward_dict)
        input_dict[DEFAULT_KEYS["likelihood_inputs"]] = self.likelihood_config(forward_dict)
        return input_dict


class DefaultLikelihoodConfigurator:
    """Fallback class for a generic configrator for amortized likelihood approximation."""

    def __init__(self, default_float_type=np.float32):
        self.default_float_type = default_float_type

    def __call__(self, forward_dict):
        """Configures the output of a generative model for likelihood estimation."""

        # Attempt to combine inputs
        input_dict = self._combine(forward_dict)

        # Convert everything to default type or fail gently
        input_dict = {k: v.astype(self.default_float_type) if v is not None else v for k, v in input_dict.items()}
        return input_dict

    def _combine(self, forward_dict):
        """Default combination for entries in forward_dict."""

        out_dict = {DEFAULT_KEYS["observables"]: None, DEFAULT_KEYS["conditions"]: None}

        # Determine whether simulated or observed data available, throw if None present
        if forward_dict.get(DEFAULT_KEYS["sim_data"]) is None and forward_dict.get(DEFAULT_KEYS["obs_data"]) is None:
            raise ConfigurationError(
                f"Either {DEFAULT_KEYS['sim_data']} or {DEFAULT_KEYS['obs_data']}"
                + " should be present as keys in the forward_dict."
            )

        # If only simulated or observed data present, all good
        elif forward_dict.get(DEFAULT_KEYS["sim_data"]) is not None:
            data = forward_dict.get(DEFAULT_KEYS["sim_data"])
        elif forward_dict.get(DEFAULT_KEYS["obs_data"]) is not None:
            data = forward_dict.get(DEFAULT_KEYS["obs_data"])

        # Else if neither 'sim_data' nor 'obs_data' present, throw again
        else:
            raise ConfigurationError(
                f"Either {DEFAULT_KEYS['sim_data']} or {DEFAULT_KEYS['obs_data']}"
                + " should be present as keys in the forward_dict."
            )

        # Extract targets and conditions
        out_dict[DEFAULT_KEYS["observables"]] = data
        out_dict[DEFAULT_KEYS["conditions"]] = forward_dict[DEFAULT_KEYS["prior_draws"]]

        return out_dict


class DefaultCombiner:
    """Fallback class for a generic combiner of conditions."""

    def __call__(self, forward_dict):
        """Converts all condition-related variables or fails."""

        out_dict = {
            DEFAULT_KEYS["summary_conditions"]: None,
            DEFAULT_KEYS["direct_conditions"]: None,
        }

        # Determine whether simulated or observed data available, throw if None present
        if forward_dict.get(DEFAULT_KEYS["sim_data"]) is None and forward_dict.get(DEFAULT_KEYS["obs_data"]) is None:
            raise ConfigurationError(
                f"Either {DEFAULT_KEYS['sim_data']} or {DEFAULT_KEYS['obs_data']}"
                + " should be present as keys in the forward_dict, but not both!"
            )

        # If only simulated or observed data present, all good
        elif forward_dict.get(DEFAULT_KEYS["sim_data"]) is not None:
            data = forward_dict.get(DEFAULT_KEYS["sim_data"])
        elif forward_dict.get(DEFAULT_KEYS["obs_data"]) is not None:
            data = forward_dict.get(DEFAULT_KEYS["obs_data"])

        # Else if neither 'sim_data' nor 'obs_data' present, throw again
        else:
            raise ConfigurationError(
                f"Either {DEFAULT_KEYS['sim_data']} or {DEFAULT_KEYS['obs_data']}"
                + " should be present as keys in the forward_dict."
            )

        # Handle simulated or observed data or throw if the data could not be converted to an array
        try:
            if type(data) is not np.ndarray:
                summary_conditions = np.array(data)
            else:
                summary_conditions = data
        except Exception as _:
            raise ConfigurationError("Could not convert input data to array...")

        # Handle prior batchable context or throw if error encountered
        if forward_dict.get(DEFAULT_KEYS["prior_batchable_context"]) is not None:
            try:
                if type(forward_dict[DEFAULT_KEYS["prior_batchable_context"]]) is not np.ndarray:
                    pbc_as_array = np.array(forward_dict[DEFAULT_KEYS["prior_batchable_context"]])
                else:
                    pbc_as_array = forward_dict[DEFAULT_KEYS["prior_batchable_context"]]
            except Exception as _:
                raise ConfigurationError("Could not convert prior batchable context to array.")

            try:
                summary_conditions = np.concatenate([summary_conditions, pbc_as_array], axis=-1)
            except Exception as _:
                raise ConfigurationError(
                    f"Could not concatenate data and prior batchable context. Shape mismatch: "
                    + "data - {summary_conditions.shape}, prior_batchable_context - {pbc_as_array.shape}."
                )

        # Handle simulation batchable context, or throw if error encountered
        if forward_dict.get(DEFAULT_KEYS["sim_batchable_context"]) is not None:
            try:
                if type(forward_dict[DEFAULT_KEYS["sim_batchable_context"]]) is not np.ndarray:
                    sbc_as_array = np.array(forward_dict[DEFAULT_KEYS["sim_batchable_context"]])
                else:
                    sbc_as_array = forward_dict[DEFAULT_KEYS["sim_batchable_context"]]
            except Exception as _:
                raise ConfigurationError("Could not convert simulation batchable context to array!")

            try:
                summary_conditions = np.concatenate([summary_conditions, sbc_as_array], axis=-1)
            except Exception as _:
                raise ConfigurationError(
                    f"Could not concatenate data (+optional prior context) and"
                    + f" simulation batchable context. Shape mismatch:"
                    + f" data - {summary_conditions.shape}, prior_batchable_context - {sbc_as_array.shape}"
                )

        # Add summary conditions to output dict
        out_dict[DEFAULT_KEYS["summary_conditions"]] = summary_conditions

        # Handle non-batchable contexts
        if (
            forward_dict.get(DEFAULT_KEYS["prior_non_batchable_context"]) is None
            and forward_dict.get(DEFAULT_KEYS["sim_non_batchable_context"]) is None
        ):
            return out_dict

        # Handle prior non-batchable context
        direct_conditions = None
        if forward_dict.get(DEFAULT_KEYS["prior_non_batchable_context"]) is not None:
            try:
                if type(forward_dict[DEFAULT_KEYS["prior_non_batchable_context"]]) is not np.ndarray:
                    pnbc_conditions = np.array(forward_dict[DEFAULT_KEYS["prior_non_batchable_context"]])
                else:
                    pnbc_conditions = forward_dict[DEFAULT_KEYS["prior_non_batchable_context"]]
            except Exception as _:
                raise ConfigurationError("Could not convert prior non_batchable_context to an array!")
            direct_conditions = pnbc_conditions

        # Handle simulation non-batchable context
        if forward_dict.get(DEFAULT_KEYS["sim_non_batchable_context"]) is not None:
            try:
                if type(forward_dict[DEFAULT_KEYS["sim_non_batchable_context"]]) is not np.ndarray:
                    snbc_conditions = np.array(forward_dict[DEFAULT_KEYS["sim_non_batchable_context"]])
                else:
                    snbc_conditions = forward_dict[DEFAULT_KEYS["sim_non_batchable_context"]]
            except Exception as _:
                raise ConfigurationError("Could not convert sim_non_batchable_context to array!")
            try:
                if direct_conditions is not None:
                    direct_conditions = np.concatenate([direct_conditions, snbc_conditions], axis=-1)
                else:
                    direct_conditions = snbc_conditions
            except Exception as _:
                raise ConfigurationError(
                    f"Could not concatenate prior non-batchable context and  \
                            simulation non-batchable context. Shape mismatch: \
                                - {direct_conditions.shape} vs. {snbc_conditions.shape}"
                )
        out_dict[DEFAULT_KEYS["direct_conditions"]] = direct_conditions
        return out_dict


class DefaultPosteriorConfigurator:
    """Fallback class for a generic configrator for amortized posterior approximation."""

    def __init__(self, default_float_type=np.float32):
        self.default_float_type = default_float_type
        self.combiner = DefaultCombiner()

    def __call__(self, forward_dict):
        """Processes the forward dict to configure the input to an amortizer."""

        # Combine inputs (conditionals)
        input_dict = self.combiner(forward_dict)
        input_dict[DEFAULT_KEYS["parameters"]] = forward_dict[DEFAULT_KEYS["prior_draws"]]

        # Convert everything to default type or fail gently
        input_dict = {k: v.astype(self.default_float_type) if v is not None else v for k, v in input_dict.items()}
        return input_dict


class DefaultModelComparisonConfigurator:
    """Fallback class for a default configurator for amortized model comparison."""

    def __init__(self, num_models, combiner=None, default_float_type=np.float32):
        self.num_models = num_models
        if combiner is None:
            self.combiner = DefaultCombiner()
        else:
            self.combiner = combiner
        self.default_float_type = default_float_type

    def __call__(self, forward_dict):
        """Convert all variables to arrays and combines them for inference into a dictionary with
        the following keys, if DEFAULT_KEYS dictionary unchanged:

        `model_indices`      - a list of model indices, e.g., if two models, then [0, 1]
        `model_outputs`      - a list of dictionaries, e.g., if two models, then [dict0, dict1]
        """

        # Prepare placeholders
        input_dict = {
            DEFAULT_KEYS["summary_conditions"]: None,
            DEFAULT_KEYS["direct_conditions"]: None,
            DEFAULT_KEYS["model_indices"]: None,
        }

        summary_conditions = []
        direct_conditions = []
        model_indices = []

        # Loop through outputs of individual models
        for m_idx, dict_m in zip(
            forward_dict[DEFAULT_KEYS["model_indices"]], forward_dict[DEFAULT_KEYS["model_outputs"]]
        ):
            # Configure individual model outputs
            conf_out = self.combiner(dict_m)

            # Extract summary conditions
            if conf_out.get(DEFAULT_KEYS["summary_conditions"]) is not None:
                summary_conditions.append(conf_out[DEFAULT_KEYS["summary_conditions"]])
                num_draws_m = conf_out[DEFAULT_KEYS["summary_conditions"]].shape[0]

            # Extract direct conditions
            if conf_out.get(DEFAULT_KEYS["direct_conditions"]) is not None:
                direct_conditions.append(conf_out[DEFAULT_KEYS["direct_conditions"]])
                num_draws_m = conf_out[DEFAULT_KEYS["direct_conditions"]].shape[0]

            model_indices.append(to_categorical([m_idx] * num_draws_m, self.num_models))

        # At this point, all elements of the input_dicts should be arrays with identical keys
        input_dict[DEFAULT_KEYS["summary_conditions"]] = (
            np.concatenate(summary_conditions) if summary_conditions else None
        )
        input_dict[DEFAULT_KEYS["direct_conditions"]] = np.concatenate(direct_conditions) if direct_conditions else None
        input_dict[DEFAULT_KEYS["model_indices"]] = np.concatenate(model_indices)

        # Convert to default types
        input_dict = {k: v.astype(self.default_float_type) if v is not None else v for k, v in input_dict.items()}
        return input_dict
