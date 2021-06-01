import numpy as np


def dm_prior(batch_size):
    """
    Samples from the prior 'batch_size' times.
    ----------

    Arguments:
    batch_size : int -- the number of samples to draw from the prior
    ----------

    Output:
    theta : np.ndarray of shape (batch_size, theta_dim) -- the samples batch of parameters
    """

    # Prior ranges for the simulator
    # v_c ~ U(-7.0, 7.0)
    # a_c ~ U(0.1, 4.0)
    # t0 ~ U(0.1, 3.0)
    p_samples = np.random.uniform(low=(0.1, 0.1, 0.1, 0.1, 0.1),
                                  high=(7.0, 7.0, 4.0, 4.0, 3.0), size=(batch_size, 5))
    return p_samples.astype(np.float32)


def model_prior(batch_size, n_models=3, p_vals=None):
    """
    Samples from the models' prior batch size times and converts to one-hot.
    Assumes equal model priors.
    ----------

    Arguments:
    batch_size : int  -- the number of samples to draw from the prior
    ----------

    Returns:
    m_true : np.ndarray of shape (batch_size, theta_dim) -- the samples batch of parameters
    """

    # Equal priors, if nothign specified
    if p_vals is None:
        p_vals = [1 / n_models] * n_models
    m_idx = np.random.choice(n_models, size=batch_size, p=p_vals).astype(np.int32)
    return m_idx


def model1_params_prior(**args):
    """
    Samples from the prior of the HH-2pars theta = (gbar_Na,gbar_K)
    ----------

    Arguments:
    ----------

    Output:
    theta : np.ndarray of shape (1, theta_dim) -- the samples of parameters
            or a dict with param key-values
    """

    theta = [
        np.random.uniform(low=1.5, high=30),
        np.random.uniform(low=0.3, high=15)
    ]
    return np.array(theta)


def model2_params_prior(**args):
    """
     Samples from the prior of the HH-3pars theta = (gbar_Na,gbar_K,gbar_M)
    ----------

    Arguments:
    ----------

    Output:
    theta : np.ndarray of shape (1, theta_dim) -- the samples of parameters
            or a dict with param key-values
    """

    theta = [
        np.random.uniform(low=1.5, high=30),
        np.random.uniform(low=0.3, high=15),
        np.random.uniform(low=0.005, high=0.3)
    ]
    return np.array(theta)


def model3_params_prior(**args):
    """
    Samples from the prior of the HH-4pars theta = (gbar_l,gbar_Na,gbar_K,gbar_M)
    ----------

    Arguments:
    ----------

    Output:
    theta : np.ndarray of shape (1, theta_dim) -- the samples of parameters
            or a dict with param key-values
    """

    theta = [
        np.random.uniform(low=0.01, high=0.18),
        np.random.uniform(low=1.5, high=30),
        np.random.uniform(low=0.1, high=15),
        np.random.uniform(low=0.005, high=0.3)
    ]
    return np.array(theta)
