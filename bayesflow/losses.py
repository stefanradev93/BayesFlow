

import tensorflow as tf

from bayesflow.computational_utilities import maximum_mean_discrepancy

def kl_latent_space_gaussian(z, log_det_J):
    """ Computes the Kullback-Leibler divergence between true and approximate
    posterior assumes a Gaussian latent space as a source distribution.

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

    >>> kl_latent_space(z, sim_data)
    """

    loss = tf.reduce_mean(0.5 * tf.math.square(tf.norm(z, axis=-1)) - log_det_J)
    return loss


def kl_latent_space_student(v, z, log_det_J):
    """ Computes the Kullback-Leibler divergence between true and approximate
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
    loss = 0.
    loss -= d * tf.math.lgamma(0.5*(v + 1))
    loss += d * tf.math.lgamma(0.5*v + 1e-15)
    loss += (0.5*d) * tf.math.log(v + 1e-15)
    loss += 0.5*(v+1) * tf.reduce_sum(tf.math.log1p(z**2 / v), axis=-1)
    loss -= log_det_J
    mean_loss = tf.reduce_mean(loss)
    return mean_loss

def mmd_summary_space(summary_outputs, z_dist=tf.random.normal):
    """ Computes the MMD(p(summary_otuputs) | z_dist) to re-shape the summary network outputs in
    an information-preserving manner.

    Parameters
    ----------
    summary_outputs   : tf Tensor of shape (batch_size, ...)
        The degrees of freedom of the latent student t-distribution

    """

    z_samples = z_dist(summary_outputs.shape) 
    mmd_loss = maximum_mean_discrepancy(summary_outputs, z_samples)
    return mmd_loss


