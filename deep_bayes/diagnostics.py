import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve


from .losses import maximum_mean_discrepancy


def calibration_error(theta_samples, theta_test, alpha_resolution=100):
    """
    Computes the calibration error of an approximate posterior per parameters.
    The calibration error is given as the median of the absolute deviation
    between alpha (0 - 1) (credibility level) and the relative number of inliers from
    theta test.
    
    ----------
    
    Arguments:
    theta_samples       : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test          : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    alpha_resolution    : int -- the number of intervals to consider 
    
    ----------
    
    Returns:
    
    cal_errs  : np.ndarray of shape (n_params, ) -- the calibration errors per parameter
    """

    n_params = theta_test.shape[1]
    n_test = theta_test.shape[0]
    alphas = np.linspace(0.01, 1.0, alpha_resolution)
    cal_errs = np.zeros(n_params)
    
    # Loop for each parameter
    for k in range(n_params):
        alphas_in = np.zeros(len(alphas))
        # Loop for each alpha
        for i, alpha in enumerate(alphas):

            # Find lower and upper bounds of posterior distribution
            region = 1 - alpha
            lower = np.round(region / 2, 3)
            upper = np.round(1 - (region / 2), 3)

            # Compute quantiles for given alpha using the entire sample
            quantiles = np.quantile(theta_samples[:, :, k], [lower, upper], axis=0).T

            # Compute the relative number of inliers
            inlier_id = (theta_test[:, k] > quantiles[:, 0]) &  (theta_test[:, k] < quantiles[:, 1])
            inliers_alpha = np.sum(inlier_id) / n_test
            alphas_in[i] = inliers_alpha
        
        # Compute calibration error for k-th parameter
        diff_alphas = np.abs(alphas - alphas_in)
        cal_err = np.round(np.median(diff_alphas), 3)
        cal_errs[k] = cal_err
        
    return cal_errs


def accuracy(m_true, m_pred):
    """
    Computes the accuracy in recovering the 'true' model by taking the argmax
    of the predicted posterior model probabilities.
    ----------
    
    Arguments:
    m_true    : np.ndarray of shape (n_test, n_models) -- the 'true' one-hot encoded model indices
    m_pred    : np.ndarray of shape (n_test, n_models) -- the predicted model posterior probabilities
    ----------
    
    Returns:
    acc  : float in [0, 1] -- the computed accuracy
    """

    # Convert tf.Tensors to numpy, if passed
    if type(m_true) is not np.ndarray:
        m_true = m_true.numpy() 
    if type(m_pred) is not np.ndarray:
        m_pred = m_pred.numpy()
    
    m_true = np.argmax(m_true, axis=1)
    m_pred = np.argmax(m_pred, axis=1)
    acc = accuracy_score(m_true, m_pred)
    return acc


def overconfidence(m_true, m_pred, overconfidence_bound=.95):
    """
    Computes the overconfidence in model selection given an overconfidence bound
    as abs(max(0, alpha_o - accuracy_o))

    Arguments:
    m_true               : np.ndarray of shape (n_test, n_models) -- the 'true' one-hot encoded model indices
    m_pred               : np.ndarray of shape (n_test, n_models) -- the predicted model posterior probabilities
    overconfidence_bound : float in [0, 1] -- the overconfidence bound (defaults to 0.95)
    ----------
    
    Returns:
    oc  : float in [0, alpha_o] -- the computed overconfidence metric
    """

    # Convert tf.Tensors to numpy, if passed
    if type(m_true) is not np.ndarray:
        m_true = m_true.numpy() 
    if type(m_pred) is not np.ndarray:
        m_pred = m_pred.numpy()
    
    pred_over_i = m_pred.max(axis=1) >= overconfidence_bound
    pred_over = m_pred.argmax(axis=1)[pred_over_i]
    m_over = m_true.argmax(axis=1)[pred_over_i]

    accuracy_over = np.sum(pred_over == m_over) / m_over.shape[0]
    return max(0, overconfidence_bound - accuracy_over)


def expected_calibration_error(m_true, m_pred, n_bins=15):
    """
    Estimates the calibration error of a model selection (classification) nn.
    Make sure that m_true are one-hot encoded classes!
    """

    # Convert tf.Tensors to numpy, if passed
    if type(m_true) is not np.ndarray:
        m_true = m_true.numpy() 
    if type(m_pred) is not np.ndarray:
        m_pred = m_pred.numpy()

    n_models = m_true.shape[1]
    cal_errs = []
    probs = []


    for k in range(n_models):

        y_true = (m_true.argmax(axis=1) == k).astype(np.float32)
        y_prob = m_pred[:, k]
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        cal_err = np.mean(np.abs(prob_true - prob_pred))
        cal_errs.append(cal_err)
        probs.append((prob_true, prob_pred))
    return cal_errs, probs


def rmse(theta_samples, theta_test, normalized=True):
    """
    Computes the RMSE or normalized RMSE (NRMSE) between posterior means 
    and true parameter values for each parameter
    
    ----------
    
    Arguments:
    theta_samples   : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test      : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    normalized      : boolean -- whether to compute nrmse or rmse (default True)
    
    ----------
    
    Returns:
    
    (n)rmse  : np.ndarray of shape (n_params, ) -- the (n)rmse per parameter
    """

    # Convert tf.Tensors to numpy, if passed
    if type(theta_samples) is not np.ndarray:
        theta_samples = theta_samples.numpy() 
    if type(theta_test) is not np.ndarray:
        theta_test = theta_test.numpy()
    
    theta_approx_means = theta_samples.mean(0)
    rmse = np.sqrt( np.mean( (theta_approx_means - theta_test)**2, axis=0) )
    
    if normalized:
        rmse = rmse / (theta_test.max(axis=0) - theta_test.min(axis=0))
    return rmse


def R2(theta_samples, theta_test):
    
    """
    Computes the R^2 score as a measure of reconstruction (percentage of variance
    in true parameters captured by estimated parameters)
    
    ----------
    Arguments:
    theta_samples   : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test      : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    
    ----------
    Returns:
    
    r2s  : np.ndarray of shape (n_params, ) -- the r2s per parameter
    """

    # Convert tf.Tensors to numpy, if passed
    if type(theta_samples) is not np.ndarray:
        theta_samples = theta_samples.numpy() 
    if type(theta_test) is not np.ndarray:
        theta_test = theta_test.numpy()
    
    theta_approx_means = theta_samples.mean(0)
    return r2_score(theta_test, theta_approx_means, multioutput='raw_values')


def resimulation_error(theta_samples, theta_test, simulator, **sim_args):
    """
    Computes the median deviation between data simulated with true true test parameters
    and data simulated with estimated parameters.
    
    ----------
    Arguments:
    theta_samples   : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test      : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    simulator       : callable -- the simulator object taking a matrix or (1, n_params) vector
                                  of parameters and returning a 3D tensor of shape (n_test, n_points, dim)
    sim_args        : arguments for the simulator
    
    ----------
    
    Returns:
    (n)rmse  : np.ndarray of shape (n_params, ) -- the (n)rmse per parameter
    """

    # Convert tf.Tensors to numpy, if passed
    if type(theta_samples) is not np.ndarray:
        theta_samples = theta_samples.numpy() 
    if type(theta_test) is not np.ndarray:
        theta_test = theta_test.numpy()
    
    theta_approx_means = theta_samples.mean(0)
    n_test = theta_test.shape[0]

    # Simulate with true and estimated
    X_test_true = simulator(theta_test, **sim_args)
    X_test_est = simulator(theta_approx_means, **sim_args)

    # Compute MMDs
    mmds = [maximum_mean_discrepancy(X_test_true[i], X_test_est[i]) for i in range(n_test)]
    return np.median(mmds)


def bootstrap_metrics(theta_samples, theta_test, simulator, p_bar=None, n_bootstrap=100, **simulator_args):
    """
    Computes bootstrap diagnostic metrics for samples from the approximate posterior.
    
    ----------
    Arguments:
    theta_samples   : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test      : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    simulator       : callable -- the simulator object taking a matrix or (1, n_params) vector
                                  of parameters and returning a 3D tensor of shape (n_test, n_points, dim)
    p_bar           : progressbar or None
    n_bootstrap     : int -- the number of bootstrap samples to take 
    simulator_args  : arguments for the simulator
    
    ----------
    
    Returns:
    (n)rmse  : np.ndarray of shape (n_params, ) -- the (n)rmse per parameter
    """
    
    n_params = int(theta_test.shape[1])
    n_test = int(theta_test.shape[0])
    
    metrics = {
        'cal_err': [],
        'rmse': [],
        'r2': [],
        'res_err': []
    }
    
    for bi in range(n_bootstrap):
        
        # Get bootstrap samples
        b_idx = np.random.choice(np.random.permutation(n_test), size=n_test, replace=True)
        theta_test_b = tf.gather(theta_test, b_idx, axis=0).numpy()
        theta_samples_b = tf.gather(theta_samples, b_idx, axis=1).numpy()
        
        # Obtain metrics on bootstrap sample
        cal_errs = calibration_error(theta_samples_b, theta_test_b)
        nrmses = rmse(theta_samples_b, theta_test_b)
        r2s = R2(theta_samples_b, theta_test_b)
        res_err = resimulation_error(theta_samples_b, theta_test_b, simulator, **simulator_args)
        
        # Add to dict
        metrics['cal_err'].append(cal_errs)
        metrics['rmse'].append(nrmses)
        metrics['r2'].append(r2s)
        metrics['res_err'].append(res_err)
        
        if p_bar is not None:
            p_bar.set_postfix_str("Bootstrap sample {}".format(bi+1))
            p_bar.update(1)
      
    # Convert to arrays for convenience
    metrics = {k: np.array(v) for k, v in metrics.items()}
    return metrics
