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

import tensorflow as tf
import numpy as np
from scipy import stats
from sklearn.calibration import calibration_curve

from bayesflow.default_settings import MMD_BANDWIDTH_LIST


def gaussian_kernel_matrix(x, y, sigmas=None):
    """Computes a Gaussian radial basis functions (RBFs) between the samples of x and y.

    We create a sum of multiple Gaussian kernels each having a width :math:`\sigma_i`.

    Parameters
    ----------
    x       :  tf.Tensor of shape (num_draws_x, num_features)
        Comprises `num_draws_x` Random draws from the "source" distribution `P`.
    y       :  tf.Tensor of shape (num_draws_y, num_features)
        Comprises `num_draws_y` Random draws from the "source" distribution `Q`.
    sigmas  : list(float), optional, default: None
        List which denotes the widths of each of the gaussians in the kernel.
        If `sigmas is None`, a default range will be used, contained in `bayesflow.default_settings.MMD_BANDWIDTH_LIST`

    Returns
    -------
    kernel  : tf.Tensor of shape (num_draws_x, num_draws_y)
        The kernel matrix between pairs from `x` and `y`.
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
    """Computes an inverse multiquadratic RBF between the samples of x and y.

    We create a sum of multiple IM-RBF kernels each having a width :math:`\sigma_i`.

    Parameters
    ----------
    x       :  tf.Tensor of shape (num_draws_x, num_features)
        Comprises `num_draws_x` Random draws from the "source" distribution `P`.
    y       :  tf.Tensor of shape (num_draws_y, num_features)
        Comprises `num_draws_y` Random draws from the "source" distribution `Q`.
    sigmas  : list(float), optional, default: None
        List which denotes the widths of each of the gaussians in the kernel.
        If `sigmas is None`, a default range will be used, contained in `bayesflow.default_settings.MMD_BANDWIDTH_LIST`

    Returns
    -------
    kernel  : tf.Tensor of shape (num_draws_x, num_draws_y)
        The kernel matrix between pairs from `x` and `y`.
    """

    if sigmas is None:
        sigmas = MMD_BANDWIDTH_LIST
    dist = tf.expand_dims(tf.reduce_sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1), axis=-1)
    sigmas = tf.expand_dims(sigmas, 0)
    return tf.reduce_sum(sigmas / (dist + sigmas), axis=-1)


def mmd_kernel(x, y, kernel):
    """Computes the estimator of the Maximum Mean Discrepancy (MMD) between two samples: x and y.

    Maximum Mean Discrepancy (MMD) is a distance-measure between random draws from 
    the distributions `x ~ P` and `y ~ Q`. 

    Parameters
    ----------
    x      : tf.Tensor of shape (N, num_features)
        An array of `N` random draws from the "source" distribution `x ~ P`.
    y      : tf.Tensor of shape (M, num_features)
        An array of `M` random draws from the "target" distribution `y ~ Q`.
    kernel : callable 
        A function which computes the distance between pairs of samples.

    Returns
    -------
    loss   : tf.Tensor of shape (,)
        The statistically biased squared maximum mean discrepancy (MMD) value.
    """

    loss = tf.reduce_mean(kernel(x, x))  
    loss += tf.reduce_mean(kernel(y, y))  
    loss -= 2 * tf.reduce_mean(kernel(x, y))  
    return loss


def mmd_kernel_unbiased(x, y, kernel):
    """Computes the unbiased estimator of the Maximum Mean Discrepancy (MMD) between two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of the distributions `x ~ P` and `y ~ Q`.

    Parameters
    ----------
    x      : tf.Tensor of shape (N, num_features)
        An array of `N` random draws from the "source" distribution `x ~ P`.
    y      : tf.Tensor of shape (M, num_features)
        An array of `M` random draws from the "target" distribution `y ~ Q`.
    kernel : callable
        A function which computes the distance between pairs of random draws from `x` and `y`.

    Returns
    -------
    loss   : tf.Tensor of shape (,)
        The statistically unbiaserd squared maximum mean discrepancy (MMD) value.
    """

    m, n = x.shape[0], y.shape[0]
    loss = (1.0/(m*(m+1))) * tf.reduce_sum(kernel(x, x))  
    loss += (1.0/(n*(n+1))) * tf.reduce_sum(kernel(y, y))  
    loss -= (2.0/(m*n)) * tf.reduce_sum(kernel(x, y))  
    return loss


def expected_calibration_error(m_true, m_pred, n_bins=15):
    """Estimates the calibration error of a model comparison network.

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


def maximum_mean_discrepancy(source_samples, target_samples, kernel='gaussian', mmd_weight=1., minimum=0.):
    """Computes the MMD given a particular choice of kernel.

    For details, consult Gretton et al. (2012):
    https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf

    Parameters
    ----------
    source_samples : tf.Tensor of shape (N, num_features)
        An array of `N` random draws from the "source" distribution.
    target_samples : tf.Tensor of shape  (M, num_features)
        An array of `M` random draws from the "target" distribution.
    kernel         : str in ('gaussian', 'inverse_multiquadratic'), optional, default: 'gaussian'
        The kernel to use for computing the distance between pairs of random draws.
    mmd_weight     : float, optional, default: 1.0
        The weight of the MMD value.
    minimum        : float, optional, default: 0.0
        The lower bound of the MMD value.

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


def get_coverage_probs(z, u):
    """Vectorized function to compute the minimal coverage probability for uniform
    ECDFs given evaluation points z and a sample of samples u.
    
    Parameters
    ----------
    z  : np.ndarray of shape (num_points, )
        The vector of evaluation points.
    u  : np.ndarray of shape (num_simulations, num_samples)
        The matrix of simulated draws (samples) from U(0, 1)
    """
    
    N = u.shape[1]
    F_m = np.sum((z[:, np.newaxis] >= u[:, np.newaxis, :] ), axis=-1) / u.shape[1]
    bin1 = stats.binom(N, z).cdf(N*F_m)
    bin2 = stats.binom(N, z).cdf(N*F_m - 1)
    gamma = 2*np.min(np.min(np.stack([bin1, 1 - bin2], axis=-1), axis=-1), axis=-1)
    return gamma


def simultaneous_ecdf_bands(num_samples, num_points=None, num_simulations=1000, 
                            confidence=0.95, eps=1e-5, max_num_points=1000):
    """Computes the simultaneous ECDF bands through simulation according to
    the algorithm described in Section 2.2:
    
    https://link.springer.com/content/pdf/10.1007/s11222-022-10090-6.pdf
    
    Depends on the vectorized utility function `get_coverage_probs(z, u)`.

    Parameters
    ----------
    num_samples     : int
        The sample size used for computing the ECDF. Will equal to the number of posterior
        samples when used for calibrarion. Corresponds to `N` in the paper above.
    num_points      : int, optional, default: None
        The number of evaluation points on the interval (0, 1). Defaults to `num_points = num_samples` if 
        not explicitly specified. Correspond to `K` in the paper above.
    num_simulations : int, optional, default: 1000
        The number of samples of size `n_samples` to simulate for determining the simultaneous CIs.
    confidence      : float in (0, 1), optional, default: 0.95
        The confidence level, `confidence = 1 - alpha` specifies the width of the confidence interval.
    eps             : float, optional, default: 1e-5
        Small number to add to the lower and subtract from the upper bound of the interval [0, 1]
        to avoid edge artefacts. No need to touch this.
    max_num_points  : int, optional, default: 1000
        Upper bound on `num_points`. Saves computation time when `num_samples` is large.
    Returns
    -------
    (alpha, z, L, U) - tuple of scalar and three arrays of size (num_samples,) containing the confidence level as well as
                       the evaluation points, the lower, and the upper confidence bands, respectively.
    """
    
    # Use shorter var names throughout
    N = num_samples
    if num_points is None:
        K = min(N, max_num_points)
    else:
        K = min(num_points, max_num_points)
    M = num_simulations
    
    # Specify evaluation points
    z = np.linspace(0+eps, 1-eps, K)
    
    # Simulate M samples of size N
    u = np.random.uniform(size=(M, N))
    
    # Get alpha
    alpha = 1 - confidence
    
    # Compute minimal coverage probabilities
    gammas = get_coverage_probs(z, u)
    
    # Use insights from paper to compute lower and upper confidence interval
    gamma = np.percentile(gammas, 100*alpha)
    L = stats.binom(N, z).ppf(gamma / 2) / N
    U = stats.binom(N, z).ppf(1 - gamma / 2) / N
    return alpha, z, L, U
