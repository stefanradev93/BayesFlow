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
import tensorflow_probability as tfp

from bayesflow.computational_utilities import maximum_mean_discrepancy


def kl_latent_space_gaussian(z, log_det_J):
    """Computes the Kullback-Leibler divergence between true and approximate
    posterior assuming a Gaussian latent space as a source distribution.

    Parameters
    ----------
    z          : tf.Tensor of shape (batch_size, ...)
        The (latent transformed) target variables
    log_det_J  : tf.Tensor of shape (batch_size, ...)
        The logartihm of the Jacobian determinant of the transformation.

    Returns
    -------
    loss : tf.Tensor
        A single scalar value representing the KL loss, shape (,)

    Examples
    --------
    Parameter estimation

    >>> kl_latent_space_gaussian(z, log_det_J)
    """

    loss = tf.reduce_mean(0.5 * tf.math.square(tf.norm(z, axis=-1)) - log_det_J)
    return loss


def kl_latent_space_student(v, z, log_det_J):
    """Computes the Kullback-Leibler divergence between true and approximate
    posterior assuming latent student t-distribution as a source distribution.

    Parameters
    ----------
    v          : tf Tensor of shape (batch_size, ...)
        The degrees of freedom of the latent student t-distribution
    z          : tf.Tensor of shape (batch_size, ...)
        The (latent transformed) target variables
    log_det_J  : tf.Tensor of shape (batch_size, ...)
        The logartihm of the Jacobian determinant of the transformation.

    Returns
    -------
    loss : tf.Tensor
        A single scalar value representing the KL loss, shape (,)
    """

    d = z.shape[-1]
    loss = 0.0
    loss -= d * tf.math.lgamma(0.5 * (v + 1))
    loss += d * tf.math.lgamma(0.5 * v + 1e-15)
    loss += (0.5 * d) * tf.math.log(v + 1e-15)
    loss += 0.5 * (v + 1) * tf.reduce_sum(tf.math.log1p(z**2 / tf.expand_dims(v, axis=-1)), axis=-1)
    loss -= log_det_J
    mean_loss = tf.reduce_mean(loss)
    return mean_loss


def kl_dirichlet(model_indices, alpha):
    """Computes the KL divergence between a Dirichlet distribution with parameter vector alpha and a uniform Dirichlet.

    Parameters
    ----------
    model_indices : tf.Tensor of shape (batch_size, n_models)
        one-hot-encoded true model indices
    alpha         : tf.Tensor of shape (batch_size, n_models)
        positive network outputs in ``[1, +inf]``

    Returns
    -------
    kl : tf.Tensor
        A single scalar representing :math:`D_{KL}(\mathrm{Dir}(\\alpha) | \mathrm{Dir}(1,1,\ldots,1) )`, shape (,)
    """

    # Extract number of models
    J = int(model_indices.shape[1])

    # Set-up ground-truth preserving prior
    alpha = alpha * (1 - model_indices) + model_indices
    beta = tf.ones((1, J), dtype=tf.float32)
    alpha0 = tf.reduce_sum(alpha, axis=1, keepdims=True)

    # Computation of KL
    kl = (
        tf.reduce_sum((alpha - beta) * (tf.math.digamma(alpha) - tf.math.digamma(alpha0)), axis=1, keepdims=True)
        + tf.math.lgamma(alpha0)
        - tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
        + tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True)
        - tf.math.lgamma(tf.reduce_sum(beta, axis=1, keepdims=True))
    )
    loss = tf.reduce_mean(kl)
    return loss


def mmd_summary_space(summary_outputs, z_dist=tf.random.normal, kernel="gaussian"):
    """Computes the MMD(p(summary_otuputs) | z_dist) to re-shape the summary network outputs in
    an information-preserving manner.

    Parameters
    ----------
    summary_outputs   : tf Tensor of shape (batch_size, ...)
        The outputs of the summary network.
    z_dist            : callable, default tf.random.normal
        The latent data distribution towards which the summary outputs are optimized.
    kernel            : str in ('gaussian', 'inverse_multiquadratic'), default 'gaussian'
        The kernel function to use for MMD computation.
    """

    z_samples = z_dist(summary_outputs.shape)
    mmd_loss = maximum_mean_discrepancy(summary_outputs, z_samples, kernel)
    return mmd_loss


def log_loss(model_indices, preds, evidential=False, label_smoothing=0.01):
    """Computes the logarithmic loss given true ``model_indices`` and approximate model
    probabilities either according to [1] if ``evidential is True`` or according to [2]
    if ``evidential is False``.

    [1] Radev, S. T., D'Alessandro, M., Mertens, U. K., Voss, A., Köthe, U., & Bürkner, P. C. (2021).
    Amortized bayesian model comparison with evidential deep learning.
    IEEE Transactions on Neural Networks and Learning Systems.

    [2] Elsemüller, L., Schnuerch, M., Bürkner, P. C., & Radev, S. T. (2023).
    A Deep Learning Method for Comparing Bayesian Hierarchical Models.
    arXiv preprint arXiv:2301.11873.

    Parameters
    ----------
    model_indices   : tf.Tensor of shape (batch_size, num_models)
        one-hot-encoded true model indices
    preds           : tf.Tensor of shape (batch_size, num_models)
        If ``evidential is True`` these should be the concentration
        parameters of a Dirichlet density bounded between ``[1, +inf]``.
        Else, these should be normalized probability values.
    evidential      : boolean, optional, default: False
        Whether to first normalize ``preds`` (True) or assume
        normalized (False, default)
    label_smoothing : float or None, optional, default: 0.01
        Optional label smoothing factor.

    Returns
    -------
    loss : tf.Tensor
        A single scalar Monte-Carlo approximation of the log-loss, shape (,)
    """

    # Apply label smoothing to indices, if specified
    if label_smoothing is not None:
        num_models = tf.cast(tf.shape(model_indices)[1], dtype=tf.float32)
        model_indices *= 1.0 - label_smoothing
        model_indices += label_smoothing / num_models

    # Obtain probs if using an evidential network
    if evidential:
        preds = preds / tf.reduce_sum(preds, axis=1, keepdims=True)

    # Numerical stability
    preds = tf.clip_by_value(preds, 1e-15, 1 - 1e-15)

    # Actual loss + regularization (if given)
    loss = -tf.reduce_mean(tf.reduce_sum(model_indices * tf.math.log(preds), axis=1))
    return loss
