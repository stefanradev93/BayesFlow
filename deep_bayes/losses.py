import tensorflow as tf
from functools import partial
import numpy as np


def maximum_likelihood_loss(z, log_det_J, **args):
    """
    Computes the ML loss as described by Ardizzone et al. (in press).
    ----------
    Arguments:
    z         : tf.Tensor of shape (batch_size, z_dim) -- the output of the final CC block f(x; c, W)
    log_det_J : tf.Tensor of shape (batch_size, )      -- the log determinant of the jacobian computed the CC block.

    Output:
    loss : tf.Tensor of shape (,)  -- a single scalar Monte-Carlo approximation of E[ ||z||^2 / 2 - log|det(J)| ]
    """

    return tf.reduce_mean(0.5 * tf.square(tf.norm(z, axis=-1)) - log_det_J)


def heteroscedastic_loss(y_true, y_mean, y_var, **args):
    """
    Computes the heteroscedastic loss for regression.

    ----------
    Arguments:
    y_true : tf.Tensor of shape (batch_size, n_out_dim) -- the vector of true values
    y_mean : tf.Tensor of shape (batch_size, n_out_dim) -- the vector fo estimated conditional means
    y_var  : tf.Tensor of shape (batch_size, n_out_dim) -- the vector of estimated conditional variance
             (alleatoric uncertainty)
    ----------
    Returns:
    loss : tf.Tensor of shape (,) -- a single scalar value representing thr heteroscedastic loss

    """

    logvar = tf.reduce_sum(0.5 * tf.log(y_var), axis=-1)
    squared_error = tf.reduce_sum(0.5 * tf.square(y_true - y_mean) / y_var, axis=-1)
    loss = tf.reduce_mean(squared_error + logvar)
    return loss


def maximum_mean_discrepancy(source_samples, target_samples, weight=1., minimum=0., **args):
    """
    This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
    different Gaussian kernels.
    ----------

    Arguments:
    x : tf.Tensor of shape  [N, num_features].
    y:  tf.Tensor of shape  [M, num_features].
    weight: the weight of the MMD loss.
    ----------

    Output:
    loss_value : tf.Tensor of shape (,) - a scalar MMD
    """

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=sigmas)
    loss_value = mmd_kernel(source_samples, target_samples, kernel=gaussian_kernel)
    loss_value = tf.maximum(minimum, loss_value) * weight
    return loss_value


def kullback_leibler_gaussian(z_mean, z_logvar, beta=1., **args):
    """
    Computes the KL divergence between a unit Gaussian and an arbitrary Gaussian.
    ----------

    Arguments:
    z_mean   : tf.Tensor of shape (batch_size, z_dim) -- the means of the Gaussian which will be compared
    z_logvar : tf.Tensor of shape (batch_size, z_dim) -- the log vars of the Gaussian to be compared
    beta     : float -- the factor to weigh the KL divergence with
    ----------

    Output:
    loss : tf.Tensor of shape (,)  -- a single scalar representing KL( N(z_mu, z_var | N(0, 1) )
    """
    
    loss = 1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
    loss = -0.5 * tf.reduce_sum(loss, axis=-1)
    return beta * tf.reduce_mean(loss)


def kullback_leibler_dirichlet(m_true, alpha):
    """
    Computes the KL divergence between a Dirichlet distribution with parameter vector alpha and a uniform Dirichlet.
    ----------

    Arguments:
    alpha : tf.Tensor of shape (batch_size, M) -- the vector of model evidences
    ----------

    Output:
    kl: tf.Tensor of shape (,)  -- a single scalar representing KL( Dir(alpha) | Dir(1,1,...,1) )
    """

    alpha = alpha * (1 - m_true) + m_true
    M = int(m_true.shape[1])
    beta = tf.constant(np.ones((1, M)), dtype=tf.float32)
    alpha0 = tf.reduce_sum(alpha, axis=1, keepdims=True)
    
    kl = tf.reduce_sum((alpha - beta) * (tf.digamma(alpha) - tf.digamma(alpha0)), axis=1, keepdims=True) + \
         tf.lgamma(alpha0) - tf.reduce_sum(tf.lgamma(alpha), axis=1, keepdims=True) + \
         tf.reduce_sum(tf.lgamma(beta), axis=1, keepdims=True) - tf.lgamma(tf.reduce_sum(beta, axis=1, keepdims=True))
    kl = tf.reduce_mean(kl)
    return kl


def kullback_leibler_iaf(z, logqz_x, beta=1., **args):
    """
    Computes the KL loss for an iaf model.
    """
    
    logpz = -tf.reduce_sum(0.5 * np.log(2*np.pi) + 0.5 * tf.square(z), axis=-1)
    kl = beta * tf.reduce_mean(logqz_x - logpz)
    return kl


def mean_squared_error(theta, theta_hat, **args):
    """
    Computes the mean squared error between two tensors.
    ----------

    Arguments:
    theta     : tf.Tensor of shape (batch_size, theta_dim) -- the true values
    theta_hat : tf.Tensor of shape (batch_size, theta_dim) -- the predicted values
    ----------

    Output:
    loss : tf.Tensor of shape (,)  -- the mean squared error
    """

    return tf.losses.mean_squared_error(theta, theta_hat)


def gaussian_kernel_matrix(x, y, sigmas):
    """
    Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    ----------

    Arguments:
    x :  tf.Tensor of shape [M, num_features]
    y :  tf.Tensor of shape [N, num_features]
    sigmas : list of floats which denotes the widths of each of the
      gaussians in the kernel.
    ----------

    Output:
    tf.Tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """

    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    dist = tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
    

def mmd_kernel(x, y, kernel=gaussian_kernel_matrix):
    """
    Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y.
    ----------

    Arguments:
    x      : tf.Tensor of shape [num_samples, num_features]
    y      : tf.Tensor of shape [num_samples, num_features]
    kernel : a function which computes the kernel in MMD. 
    ----------

    Output:
    loss : tf.Tensor of shape (,) denoting the squared maximum mean discrepancy loss.
    """

    loss = tf.reduce_mean(kernel(x, x))
    loss += tf.reduce_mean(kernel(y, y))
    loss -= 2 * tf.reduce_mean(kernel(x, y))
    return loss


def bayes_risk(m_true, alpha, alpha0, m_probs, **args):
    """
    Computes the Bayes risk with respect to a Dirichlet posterior.
    ----------

    Arguments:
    m_true    : tf.Tensor of shape (batch_size, num_models) -- the one hot encoded true model indices
    alpha     : tf.Tensor of shape (batch_size, num_models) -- the model evidences 
    alpha0    : tf.Tensor of shape (batch_size, 1) -- the Dirichlet strength 
    m_probs   : tf.Tensor of shape (batch_size, num_models) -- the posterior model probabilities
    ----------

    Output:
    risk : tf.Tensor of shape (,) -- a single scalar Monte-Carlo approximation of the Bayes risk
    """

    pred_mean = tf.reduce_sum((m_true - m_probs)**2, axis=1, keepdims=True)
    pred_var = tf.reduce_sum(alpha * (alpha0 - alpha) / (alpha0 * alpha0 * (alpha0 + 1)), axis=1, keepdims=True)
    risk = tf.reduce_mean(pred_mean + pred_var)
    return risk


def regularized_bayes_risk(m_true, alpha, alpha0, m_probs, global_step, annealing_step=1000, max_lambda=1.0):
    """
    Computes the Bayes risk with respect to a Dirichlet posterior (regularized via KL)
    ----------

    Arguments:
    m_true    : tf.Tensor of shape (batch_size, num_models) -- the one hot encoded true model indices
    alpha     : tf.Tensor of shape (batch_size, num_models) -- the model evidences 
    alpha0    : tf.Tensor of shape (batch_size, 1) -- the Dirichlet strength 
    m_probs   : tf.Tensor of shape (batch_size, num_models) -- the posterior model probabilities
    ----------

    Output:
    risk : tf.Tensor of shape (,) -- a single scalar Monte-Carlo approximation of the regularized Bayes risk
    """

    risk = bayes_risk(m_true, alpha, alpha0, m_probs)
    kl = kullback_leibler_dirichlet(m_true, alpha)
    lamb = tf.cast(tf.minimum(max_lambda, global_step / annealing_step), dtype=tf.float32)
    loss = risk + lamb * kl
    return loss


def heteroscedastic_loglik(x, m_true):
    """
    Computes the E[p] w.r.t. a Gaussian N(x_mean, x_var).
    ----------

    Arguments:
    x         : tf.Tensor of shape (batch_size, num_models) -- the noisy logits
    m_true    : tf.Tensor of shape (batch_size, num_models) -- the one hot encoded true model indices
    
    ----------

    Output:
    ll : tf.Tensor of shape (,) -- a single scalar Monte-Carlo approximation of the heteroscedastic loss
    """
    
    logsumexp = tf.log(tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True) + 1e-20)
    ll = x - logsumexp
    ll = tf.boolean_mask(ll, m_true)
    ll = tf.reduce_mean(ll)
    return ll
    

def log_loss(m_true, alpha, alpha0, m_probs, lambd=1.0):
    """
    Computes the logloss given output probs and true model indices m_true.
    ----------

    Arguments:
    m_true    : tf.Tensor of shape (batch_size, num_models) -- the one hot encoded true model indices
    alpha     : tf.Tensor of shape (batch_size, num_models) -- the model evidences 
    alpha0    : tf.Tensor of shape (batch_size, 1) -- the Dirichlet strength 
    m_probs   : tf.Tensor of shape (batch_size, num_models) -- the posterior model probabilities
    lambd     : float in (0, 1) -- the weight of the KL regularization term
    ----------

    Output:
    loss : tf.Tensor of shape (,) -- a single scalar Monte-Carlo approximation of the regularized Bayes risk
    """
    
    m_probs = tf.clip_by_value(m_probs, 1e-15, 1 - 1e-15)
    loss = -tf.reduce_mean(tf.reduce_sum(m_true * tf.log(m_probs), axis=1))
    if lambd > 0:
        kl = kullback_leibler_dirichlet(m_true, alpha)
        loss = loss + lambd * kl
    return loss

def brier_score(m_true, alpha, alpha0, m_probs):
    """
    Computes the Brier score given output probs and true model indices.
    ----------

    Arguments:
    m_true    : tf.Tensor of shape (batch_size, num_models) -- the one hot encoded true model indices
    alpha     : tf.Tensor of shape (batch_size, num_models) -- the model evidences 
    alpha0    : tf.Tensor of shape (batch_size, 1) -- the Dirichlet strength 
    m_probs   : tf.Tensor of shape (batch_size, num_models) -- the posterior model probabilities
    ----------

    Output:
    loss : tf.Tensor of shape (,) -- a single scalar Monte-Carlo approximation of the regularized Bayes risk
    """
    
    score = 1 + tf.reduce_sum(m_probs**2, axis=-1) - 2 * tf.reduce_sum(m_true * m_probs, axis=-1)
    m_score = tf.reduce_mean(score)
    return m_score



def cross_entropy(m_true, alpha, alpha0, m_probs, lambd=1.0):
    """
    Computes the Bayes risk with respect to the cross entropy loss.
    ----------

    Arguments:
    m_true    : tf.Tensor of shape (batch_size, num_models) -- the one hot encoded true model indices
    alpha     : tf.Tensor of shape (batch_size, num_models) -- the model evidences 
    alpha0    : tf.Tensor of shape (batch_size, 1) -- the Dirichlet strength 
    m_probs   : tf.Tensor of shape (batch_size, num_models) -- the posterior model probabilities
    ----------

    Output:
    loss : tf.Tensor of shape (,) -- a single scalar Monte-Carlo approximation of the cross entropy
    """

    loss = tf.reduce_sum(m_true * (tf.digamma(alpha0) - tf.digamma(alpha)), 1, keepdims=True)
    loss = tf.reduce_mean(loss)
    if lambd > 0:
        kl = kullback_leibler_dirichlet(m_true, alpha)
        loss = loss + lambd * kl
    return loss




def multinomial_likelihood(m_true, alpha, alpha0, m_probs):
    """
    Computes the type II likelihood with a Dirichlet prior.
    ----------

    Arguments:
    m_true    : tf.Tensor of shape (batch_size, num_models) -- the one hot encoded true model indices
    alpha     : tf.Tensor of shape (batch_size, num_models) -- the model evidences 
    alpha0    : tf.Tensor of shape (batch_size, 1) -- the Dirichlet strength 
    m_probs   : tf.Tensor of shape (batch_size, num_models) -- the posterior model probabilities
    ----------

    Output:
    ll : tf.Tensor of shape (,) -- a single scalar Monte-Carlo approximation of the type II ML
    """

    ll = tf.reduce_sum(m_true * (tf.log(alpha0) - tf.log(alpha)), 1, keepdims=True)
    ll = tf.reduce_mean(ll)
    return ll


def regularized_multinomial_likelihood(m_true, alpha, alpha0, m_probs, global_step, annealing_step=1000, max_lambda=1.0):
    """
    Computes the type II likelihood with a Dirichlet prior.
    ----------

    Arguments:
    m_true    : tf.Tensor of shape (batch_size, num_models) -- the one hot encoded true model indices
    alpha     : tf.Tensor of shape (batch_size, num_models) -- the model evidences 
    alpha0    : tf.Tensor of shape (batch_size, 1) -- the Dirichlet strength 
    m_probs   : tf.Tensor of shape (batch_size, num_models) -- the posterior model probabilities
    ----------

    Output:
    loss : tf.Tensor of shape (,) -- a single scalar Monte-Carlo approximation of the type II ML
    """

    ll = multinomial_likelihood(m_true, alpha, alpha0, m_probs)
    kl = kullback_leibler_dirichlet(m_true, alpha)
    lamb = tf.cast(tf.minimum(max_lambda, global_step / annealing_step), dtype=tf.float32)
    loss = ll + lamb * kl
    return loss