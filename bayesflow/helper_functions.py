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

import copy
import tensorflow as tf

from bayesflow import default_settings
from bayesflow.exceptions import ConfigurationError


def apply_gradients(optimizer, gradients, tensors, clip_value, clip_method):
        """Updates each tensor in the 'variables' list via backpropagation. Operation is performed in-place.

        Parameters
        ----------
        optimizer        : tf.keras.optimizer.Optimizer 
            Optimizer for the neural network. 
        gradients        : list(tf.Tensor)
            The list of gradients for all neural network parameters
        tensors          : list(tf.Tensor)
            The list of all neural network parameters
        clip_method      : {'norm', 'value', 'global_norm'}
            Optional gradient clipping method
        clip_value       : float
            The value used for gradient clipping when clip_method is in {'value', 'norm'}
        """

        # Optional gradient clipping
        if clip_value is not None:
            gradients = clip_gradients(gradients, clip_value=clip_value, clip_method=clip_method)
        optimizer.apply_gradients(zip(gradients, tensors))


def clip_gradients(gradients, clip_value=5., clip_method='norm'):
    """ Performs gradient clipping on a list of gradients.

    This function clips gradients by one of the following methods:

    -  value clipping,
    -  norm clipping or
    -  global norm clipping.

    Parameters
    ----------
    gradients: list(tf.Tensor)
        The computed gradients for neural network parameters.
    clip_value: float > 0
        The value used for clipping.
    clip_method: {'norm', 'global_norm', 'value'}
        The method used for clipping.

    Returns
    -------
    gradients: list
        The clipped gradients

    Raises
    ------
    ValueError
        If an unknown clipping method is specified.
    """

    if clip_method == 'global_norm':
        gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
    elif clip_method == 'norm':
        gradients = [tf.clip_by_norm(grad, clip_value) for grad in gradients if grad is not None]
    elif clip_method == 'value':
        gradients = [tf.clip_by_value(grad, -clip_value, clip_value) for grad in gradients if grad is not None]
    else:
        raise ValueError("clip_method parameter should be a string in ['norm', 'global_norm', 'value']")
    return gradients


def merge_left_into_right(left_dict, right_dict):
    """ Function to merge nested dict `left_dict` into nested dict `right_dict`.
    """
    for k, v in left_dict.items():
        if isinstance(v, dict):
            right_dict[k] = merge_left_into_right(v, right_dict[k])
        else:
            right_dict[k] = v
    return right_dict


def build_meta_dict(user_dict: dict, default_setting: default_settings.MetaDictSetting) -> dict:
    """ Integrates a user-defined dictionary into a default dictionary.

    Takes a user-defined dictionary and a default dictionary.

    #. Scan the `user_dict` for violations by unspecified mandatory fields.
    #. Merge `user_dict` entries into the `default_dict`. Considers nested dict structure.

    Parameters
    ----------
    user_dict: dict
        The user's dictionary
    default_setting: MetaDictSetting
        The specified default setting with attributes:

        -  `meta_dict`: dictionary with default values.
        -  `mandatory_fields`: list(str) keys that need to be specified by the `user_dict`

    Returns
    -------
    merged_dict: dict
        Merged dictionary

    """

    default_dict = copy.deepcopy(default_setting.meta_dict)
    mandatory_fields = copy.deepcopy(default_setting.mandatory_fields)

    # Check if all mandatory fields are provided by the user
    if not all([field in user_dict.keys() for field in mandatory_fields]):
        raise ConfigurationError(f"Not all mandatory fields provided! Need at least the following: {mandatory_fields}")

    # Merge the user dict into the default dict
    merged_dict = merge_left_into_right(user_dict, default_dict)
    return merged_dict
