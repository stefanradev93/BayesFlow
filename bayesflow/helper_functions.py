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
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from bayesflow import default_settings
from bayesflow.exceptions import ConfigurationError, ShapeError


def check_tensor_sanity(tensor, logger):
    """Tests for the presence of NaNs and Infs in a tensor."""

    if tf.executing_eagerly():
        if tf.reduce_any(tf.math.is_nan(tensor)):
            num_na = tf.reduce_sum(tf.cast(tf.math.is_nan(tensor), tf.int8)).numpy()
            logger.warn(f"Warning! Returned estimates contain {num_na} nan values!")
        if tf.reduce_any(tf.math.is_inf(tensor)):
            num_inf = tf.reduce_sum(tf.cast(tf.math.is_inf(tensor), tf.int8)).numpy()
            logger.warn(f"Warning! Returned estimates contain {num_inf} inf values!")
    else:
        if tf.reduce_any(tf.math.is_nan(tensor)):
            num_na = tf.reduce_sum(tf.cast(tf.math.is_nan(tensor), tf.int8))
            tf.print("Warning! Returned estimates contain", num_na, "nan values!")
        if tf.reduce_any(tf.math.is_inf(tensor)):
            num_inf = tf.reduce_sum(tf.cast(tf.math.is_inf(tensor), tf.int8))
            tf.print(f"Warning! Returned estimates contain", num_inf, "inf values!")


def merge_left_into_right(left_dict, right_dict):
    """Function to merge nested dict `left_dict` into nested dict `right_dict`."""
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
    """Integrates a user-defined dictionary into a default dictionary.

    Takes a user-defined dictionary and a default dictionary.

    #. Scan the `user_dict` for violations by unspecified mandatory fields.
    #. Merge `user_dict` entries into the `default_dict`. Considers nested dict structure.

    Parameters
    ----------
    user_dict       : dict
        The user's dictionary
    default_setting : MetaDictSetting
        The specified default setting with attributes:

        -  `meta_dict`: dictionary with default values.
        -  `mandatory_fields`: list(str) keys that need to be specified by the `user_dict`

    Returns
    -------
    merged_dict: dict
        Merged dictionary.
    """

    default_dict = copy.deepcopy(default_setting.meta_dict)
    mandatory_fields = copy.deepcopy(default_setting.mandatory_fields)

    # Check if all mandatory fields are provided by the user
    if not all([field in user_dict.keys() for field in mandatory_fields]):
        raise ConfigurationError(f"Not all mandatory fields provided! Need at least the following: {mandatory_fields}")

    # Merge the user dict into the default dict
    merged_dict = merge_left_into_right(user_dict, default_dict)
    return merged_dict


def extract_current_lr(optimizer):
    """Extracts current learning rate from `optimizer`.

    Parameters
    ----------
    optimizer  : instance of subclass of `tf.keras.optimizers.Optimizer`
        Optimizer to extract the learning rate from

    Returns
    -------
    current_lr : np.float or NoneType
        Current learning rate, or `None` if it can't be determined
    """

    if isinstance(optimizer.lr, LearningRateSchedule):
        # LearningRateSchedule instances need number of iterations
        current_lr = optimizer.lr(optimizer.iterations).numpy()
    elif hasattr(optimizer.lr, "numpy"):
        # Convert learning rate to numpy
        current_lr = optimizer.lr.numpy()
    else:
        # Unable to extract numerical value from optimizer.lr
        current_lr = None
    return current_lr


def format_loss_string(
    ep, it, loss, avg_dict, slope=None, lr=None, ep_str="Epoch", it_str="Iter", scalar_loss_str="Loss"
):
    """Prepare loss string for displaying on progress bar."""

    # Prepare info part
    disp_str = f"{ep_str}: {ep}, {it_str}: {it}"
    if type(loss) is dict:
        for k, v in loss.items():
            disp_str += f",{k}: {v.numpy():.3f}"
    else:
        disp_str += f",{scalar_loss_str}: {loss.numpy():.3f}"
    # Add running
    if avg_dict is not None:
        for k, v in avg_dict.items():
            disp_str += f",{k}: {v:.3f}"
    if slope is not None:
        disp_str += f",L.Slope: {slope:.3f}"
    if lr is not None:
        disp_str += f",LR: {lr:.2E}"
    return disp_str


def loss_to_string(ep, loss, ep_str="Epoch", scalar_loss_str="Loss"):
    """Converts output from an amortizer into a string.
    For instance, if a ``dict`` is provided, it will be converted as, e.g.,:
    dictionary = {k1: v1, k2: v2} -> 'k1: v1, k2: v2'
    """

    disp_str = f"Validation, {ep_str}: {ep}"
    if type(loss) is dict:
        for k, v in loss.items():
            disp_str += f", {k}: {v.numpy():.3f}"
    else:
        disp_str += f", {scalar_loss_str}: {loss.numpy():.3f}"
    return disp_str


def backprop_step(input_dict, amortizer, optimizer, **kwargs):
    """Computes the loss of the provided amortizer given an input dictionary and applies gradients.

    Parameters
    ----------
    input_dict  : dict
        The configured output of the genrative model
    amortizer   : tf.keras.Model
        The custom amortizer. Needs to implement a compute_loss method.
    optimizer   : tf.keras.optimizers.Optimizer
        The optimizer used to update the amortizer's parameters.
    **kwargs    : dict
        Optional keyword arguments passed to the network's compute_loss method

    Returns
    -------
    loss : dict
        The outputs of the compute_loss() method of the amortizer comprising all
        loss components, such as divergences or regularization.
    """

    # Forward pass and loss computation
    with tf.GradientTape() as tape:
        # Compute custom loss
        loss = amortizer.compute_loss(input_dict, training=True, **kwargs)
        # If dict, add components
        if type(loss) is dict:
            _loss = tf.add_n(list(loss.values()))
        else:
            _loss = loss
        # Collect regularization loss, if any
        if amortizer.losses != []:
            reg = tf.add_n(amortizer.losses)
            _loss += reg
            if type(loss) is dict:
                loss["W.Decay"] = reg
            else:
                loss = {"Loss": loss, "W.Decay": reg}
    # One step backprop and return loss
    gradients = tape.gradient(_loss, amortizer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, amortizer.trainable_variables))
    return loss


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
        raise ShapeError(
            f"post_samples should be a 3-dimensional array, with the "
            + f"first dimension being the number of (simulated) data sets, "
            + f"the second dimension being the number of posterior draws per data set, "
            + f"and the third dimension being the number of parameters (marginal distributions), "
            + f"but your input has dimensions {len(post_samples.shape)}"
        )
    elif len(prior_samples.shape) != 2:
        raise ShapeError(
            f"prior_samples should be a 2-dimensional array, with the "
            + f"first dimension being the number of (simulated) data sets / prior draws "
            + f"and the second dimension being the number of parameters (marginal distributions), "
            + f"but your input has dimensions {len(prior_samples.shape)}"
        )
    elif post_samples.shape[0] != prior_samples.shape[0]:
        raise ShapeError(
            f"The number of elements over the first dimension of post_samples and prior_samples"
            + f"should match, but post_samples has {post_samples.shape[0]} and prior_samples has "
            + f"{prior_samples.shape[0]} elements, respectively."
        )
    elif post_samples.shape[-1] != prior_samples.shape[-1]:
        raise ShapeError(
            f"The number of elements over the last dimension of post_samples and prior_samples"
            + f"should match, but post_samples has {post_samples.shape[1]} and prior_samples has "
            + f"{prior_samples.shape[-1]} elements, respectively."
        )
