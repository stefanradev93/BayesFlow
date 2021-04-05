import tensorflow as tf

from bayesflow import default_settings
from bayesflow.exceptions import ConfigurationError


def clip_gradients(gradients, clip_value=5., clip_method='norm'):
    """
    Peforms gradient clipping on a list of gradients by using either value
    clipping, norm clipping or global norm clipping. Raises ValueError if
    an unknown clipping method is specified.
    ----------

    Arguments:
    gradients: list of tf.Tensor -- the computed gradients for neural network parameter
    ----------

    Keyword Arguments:
    clip_value: float > 0 -- the value used for clipping 
    clip_method: str -- the method used for clipping, either 'norm', 'global_norm', or 'value'
    ----------

    Returns:
    gradients: list -- the clipped gradients
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
    """
    Function to merge nested dict left_dict into nested dict right_dict
    """
    for k, v in left_dict.items():
        if isinstance(v, dict):
            right_dict[k] = merge_left_into_right(v, right_dict[k])
        else:
            right_dict[k] = v
    return right_dict


def build_meta_dict(user_dict: dict, default_setting: default_settings.MetaDictSetting) -> dict:
    """
    Takes a user-defined dictionary and a default dictionary.
    Firstly, scans the user_dict for violations by unspecified mandatory fields.
    Secondly, merges user dict entries into the default_dict. Considers nested dict structure.
    ----------

    Arguments:
    user_dict: dict -- the user's dictionary
    default_setting: MetaDictSetting -- the specified default setting, consisting of:
                        meta_dict: dictionary with default values.
                        mandatory_fields: list of str -- keys that need to be specified by the user_dict
    ----------

    Returns:
    merged dict

    """

    default_dict = default_setting.meta_dict
    mandatory_fields = default_setting.mandatory_fields

    # Check if all mandatory fields are provided by the user
    if not all([field in user_dict.keys() for field in mandatory_fields]):
        raise ConfigurationError(f"Not all mandatory fields provided! Need at least the following: {mandatory_fields}")

    # Merge the user dict into the default dict
    return merge_left_into_right(user_dict, default_dict)
