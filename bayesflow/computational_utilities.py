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

import tensorflow as tf
import numpy as np
from sklearn.calibration import calibration_curve

from bayesflow.default_settings import MMD_BANDWIDTH_LIST


def gaussian_kernel_matrix(x, y, sigmas=None):
    """ Computes a Gaussian Radial Basis Kernel between the samples of x and y.

    We create a sum of multiple Gaussian kernels each having a width :math:`\sigma_i`.

    Parameters
    ----------
    x      :  tf.Tensor of shape (N, num_features)
    y      :  tf.Tensor of shape (M, num_features)
    sigmas :  list(float), optional, default: None (use default)
        List which denotes the widths of each of the Gaussians in the kernel.

    Returns
    -------
    kernel : tf.Tensor
        RBF kernel of shape [num_samples{x}, num_samples{y}]
    """

    if sigmas is None:
        sigmas = MMD_BANDWIDTH_LIST
    norm = lambda v: tf.reduce_sum(tf.square(v), 1)
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    kernel = tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
    return kernel


def inverse_multiquadratic_kernel_matrix(x, y, sigmas=None):
    """ Computes an inverse multiquadratic RBF between the samples of x and y.

    We create a sum of multiple IM-RBF kernels each having a width :math:`\sigma_i`.

    Parameters
    ----------
    x :  tf.Tensor of shape (N, num_features)
    y :  tf.Tensor of shape (M, num_features)
    sigmas : list(float)
        List which denotes the widths of each of the gaussians in the kernel.

    Returns
    -------
    kernel: tf.Tensor
        RBF kernel of shape [num_samples{x}, num_samples{y}]
    """

    if sigmas is None:
        sigmas = MMD_BANDWIDTH_LIST
    dist = tf.expand_dims(tf.reduce_sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1), axis=-1)
    sigmas = tf.expand_dims(sigmas, 0)
    return tf.reduce_sum(sigmas / (dist + sigmas), axis=-1)


def mmd_kernel(x, y, kernel):
    """ Computes the Maximum Mean Discrepancy (MMD) between two samples: x and y.

    Maximum Mean Discrepancy (MMD) is a distance-measure between random draws from 
    the distributions of x and y.

    Parameters
    ----------
    x      : tf.Tensor of shape (N, num_features)
        Random dtaws 
    y      : tf.Tensor of shape (M, num_features)
    kernel : callable 
        A function which computes the kernel for MMD.

    Returns
    -------
    loss   : tf.Tensor
        squared maximum mean discrepancy loss, shape (,)
    """

    loss = tf.reduce_mean(kernel(x, x))  
    loss += tf.reduce_mean(kernel(y, y))  
    loss -= 2 * tf.reduce_mean(kernel(x, y))  
    return loss


def mmd_kernel_unbiased(x, y, kernel):
    """ Computes the unbiased estimator of the Maximum Mean Discrepancy (MMD) between two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of the distributions of x and y.

    Parameters
    ----------
    x      : tf.Tensor of shape (N, num_features)
    y      : tf.Tensor of shape (M, num_features)
    kernel : callable
        A function which computes the kernel for MMD.

    Returns
    -------
    loss   : tf.Tensor
        squared maximum mean discrepancy loss, shape (,)
    """

    m, n = x.shape[0], y.shape[0]
    loss = (1.0/(m*(m+1))) * tf.reduce_sum(kernel(x, x))  
    loss += (1.0/(n*(n+1))) * tf.reduce_sum(kernel(y, y))  
    loss -= (2.0/(m*n)) * tf.reduce_sum(kernel(x, y))  
    return loss


def expected_calibration_error(m_true, m_pred, n_bins=15):
    """ Estimates the calibration error of a model comparison network.

    Important
    ---------
    Make sure that ``m_true`` are **one-hot encoded** classes!

    Parameters
    ----------
    m_true  : np.array or list
        True model indices
    m_pred  : np.array or list
        Predicted model indices
    n_bins  : int, default: 15
        Number of bins for plot

    Returns
    -------
    #TODO
    """

    # Convert tf.Tensors to numpy, if passed
    if type(m_true) is not np.ndarray:
        m_true = m_true.numpy() 
    if type(m_pred) is not np.ndarray:
        m_pred = m_pred.numpy()
    
    # Extract number of models and prepare containers
    n_models = m_true.shape[1]
    cal_errs = []
    probs = []

    # Loop for each model and compute calibration errs per bin
    for k in range(n_models):

        y_true = (m_true.argmax(axis=1) == k).astype(np.float32)
        y_prob = m_pred[:, k]
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        cal_err = np.mean(np.abs(prob_true - prob_pred))
        cal_errs.append(cal_err)
        probs.append((prob_true, prob_pred))
    return cal_errs, probs


def maximum_mean_discrepancy(source_samples, target_samples, kernel, mmd_weight=1., minimum=0.):
    """ Computes the MMD given a particular choice of kernel.

    Parameters
    ----------
    source_samples :  tf.Tensor of shape (N, num_features)
    target_samples :  tf.Tensor of shape  (M, num_features)
    kernel         : str in ('gaussian', 'inverse_multiquadratic')
    mmd_weight     : float, default: 1.0
        the weight of the MMD loss.
    minimum        : float, default: 0.0
        lower loss bound

    Returns
    -------
    loss_value : tf.Tensor
        A scalar Maximum Mean Discrepancy, shape (,)
    """

    # Determine kernel, fall back to Gaussian if unknown string passed
    if kernel == "gaussian":
        kernel_fun = gaussian_kernel_matrix
    elif kernel == "inverse_multiquadratic":
        kernel_fun = inverse_multiquadratic_kernel_matrix
    else:
        kernel_fun = gaussian_kernel_matrix
    # Compute and return MMD value
    loss_value = mmd_kernel(source_samples, target_samples, kernel=kernel_fun)
    loss_value = mmd_weight * tf.maximum(minimum, loss_value)
    return loss_value


