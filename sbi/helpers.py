import tensorflow as tf



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

