"""
This module implements utilities for computing, clipping gradients or training deep learning models.
"""


__version__ = '0.1'
__author__ = 'Stefan Radev'


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

    if clip_method == 'norm':
        gradients = [tf.clip_by_norm(grad, clip_value) for grad in gradients if grad is not None]

    elif clip_method == 'global_norm':
        gradients, _ = tf.clip_by_global_norm(gradients, clip_value)

    elif clip_method == 'value':
        gradients = [tf.clip_by_value(grad, -clip_value, clip_value) for grad in gradients if grad is not None]

    else:
        raise ValueError("clip_method parameter should be a string in ['norm', 'global_norm', 'value']")

    return gradients

        
def apply_gradients(optimizer, gradients, variables, global_step=None):
    """
    Performs one step of the backprop algorithm by updating each tensor in the 'variables' list.
    Note, that the opertaion is performed in-place.
    ----------

    Arguments:
    optimizer: tf.train.Optimizer -- an optimizer instance supporting an apply_gradeints() method
    gradients: list of tf.Tensor -- the list of gradients for all neural network parameter
    variables: list of tf.Tensor -- the list of all neural network parameters
    ----------

    Keyword Arguments:
    clip_value: global_step: tf.Variable -- an integer valued tf.Variable indicating the current iteration step 
    ----------
    """

    optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)