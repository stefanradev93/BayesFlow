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

import copy
import tensorflow as tf

from bayesflow import default_settings
from bayesflow.exceptions import ConfigurationError, ShapeError


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
    gradients: list of tf.Tensor
        The computed gradients of neural network parameters.
    clip_value: float > 0
        The value used for clipping.
    clip_method: {'norm', 'global_norm', 'value'}
        The method used for clipping.

    Returns
    -------
    gradients: list of tf.Tensor
        The clipped gradients as given by the specified method.

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
            if right_dict.get(k) is not None:
                right_dict[k] = merge_left_into_right(v, right_dict.get(k))
            else:
                right_dict[k] = v
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


def format_loss_string(ep, it, loss, avg_dict, slope, ep_str="Epoch", it_str='Iter', scalar_loss_str='Loss'):
    """ Prepare loss string for displaying on progress bar
    """

    disp_str = f"{ep_str}: {ep}, {it_str}: {it}"
    if type(loss) is dict:
        for k, v in loss.items():
            disp_str += f",{k}: {v.numpy():.3f}"
    else:
        disp_str  += f",{scalar_loss_str}: {loss.numpy():.3f}"
    # Add running
    if avg_dict is not None:
        for k, v in avg_dict.items():
            disp_str += f",{k}: {v:.3f}"
    if slope is None:
        disp_str += f",L.Slope: NA"
    else:
        disp_str += f",L.Slope: {slope:.3f}"
    return disp_str


def check_posterior_prior_shapes(post_samples, prior_samples):
    """Checks requirements for the shapes of posterior and prior draws as
    necessitated by most diagnostic functions.
    
    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws obtained for generating n_data_sets

    Raises
    ------
    ShapeError 
        If there is a deviation form the expected shapes of `post_samples` and `prior_samples`.
    """

    if len(post_samples.shape) != 3:
        raise ShapeError(f'post_samples should be a 3-dimensional array, with the ' +
                         f'first dimension being the number of (simulated) data sets, ' + 
                         f'the second dimension being the number of posterior draws per data set, ' + 
                         f'and the third dimension being the number of parameters (marginal distributions), ' +  
                         f'but your input has dimensions {len(post_samples.shape)}')
    elif len(prior_samples.shape) != 2:
        raise ShapeError(f'prior_samples should be a 2-dimensional array, with the ' +  
                         f'first dimension being the number of (simulated) data sets / prior draws ' + 
                         f'and the second dimension being the number of parameters (marginal distributions), ' +  
                         f'but your input has dimensions {len(prior_samples.shape)}')
    elif post_samples.shape[0] != prior_samples.shape[0]:
        raise ShapeError(f'The number of elements over the first dimension of post_samples and prior_samples' + 
                         f'should match, but post_samples has {post_samples.shape[0]} and prior_samples has ' +
                         f'{prior_samples.shape[0]} elements, respectively.')
    elif post_samples.shape[-1] != prior_samples.shape[-1]:
        raise ShapeError(f'The number of elements over the last dimension of post_samples and prior_samples' + 
                         f'should match, but post_samples has {post_samples.shape[1]} and prior_samples has ' +
                         f'{prior_samples.shape[-1]} elements, respectively.')
