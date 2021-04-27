import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import binom
from sklearn.calibration import calibration_curve
from sklearn.metrics import r2_score, confusion_matrix


def true_vs_estimated(theta_true, theta_est, param_names, dpi=300,
                      figsize=(20, 4), show=True, filename=None, font_size=12):
    """ Plots a scatter plot with abline of the estimated posterior means vs true values.

    Parameters
    ----------
    theta_true: np.array
        Array of true parameters.
    theta_est: np.array
        Array of estimated parameters.
    param_names: list(str)
        List of parameter names for plotting.
    dpi: int, default:300
        Dots per inch (dpi) for the plot.
    figsize: tuple(int, int), default: (20,4)
        Figure size.
    show: boolean, default: True
        Controls if the plot will be shown
    filename: str, default: None
        Filename if plot shall be saved
    font_size: int, default: 12
        Font size

    """


    # Plot settings
    plt.rcParams['font.size'] = font_size

    # Determine n_subplots dynamically
    n_row = int(np.ceil(len(param_names) / 6))
    n_col = int(np.ceil(len(param_names) / n_row))

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat
        
    # --- Plot true vs estimated posterior means on a single row --- #
    for j in range(len(param_names)):
        
        # Plot analytic vs estimated
        axarr[j].scatter(theta_est[:, j], theta_true[:, j], color='black', alpha=0.4)
        
        # get axis limits and set equal x and y limits
        lower_lim = min(axarr[j].get_xlim()[0], axarr[j].get_ylim()[0])
        upper_lim = max(axarr[j].get_xlim()[1], axarr[j].get_ylim()[1])
        axarr[j].set_xlim((lower_lim, upper_lim))
        axarr[j].set_ylim((lower_lim, upper_lim))
        axarr[j].plot(axarr[j].get_xlim(), axarr[j].get_xlim(), '--', color='black')
        
        # Compute NRMSE
        rmse = np.sqrt(np.mean( (theta_est[:, j] - theta_true[:, j])**2 ))
        nrmse = rmse / (theta_true[:, j].max() - theta_true[:, j].min())
        axarr[j].text(0.1, 0.9, 'NRMSE={:.3f}'.format(nrmse),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes,
                     size=10)
        
        # Compute R2
        r2 = r2_score(theta_true[:, j], theta_est[:, j])
        axarr[j].text(0.1, 0.8, '$R^2$={:.3f}'.format(r2),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes, 
                     size=10)
        
        if j == 0:
            # Label plot
            axarr[j].set_xlabel('Estimated')
            axarr[j].set_ylabel('True')
        axarr[j].set_title(param_names[j])
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)
    
    # Adjust spaces
    f.tight_layout()

    if show:
        plt.show()

    if filename is not None:
        f.savefig(filename)


def plot_sbc(theta_samples, theta_test, param_names, bins=25, dpi=300,
            figsize=(24, 12), interval=0.99, show=True, font_size=12):
    """ Plots the simulation-based posterior checking histograms as advocated by Talts et al. (2018).

    Parameters
    ----------
    theta_samples: np.array
        Array of sampled parameters
    theta_test: np.array
        Array of test parameters
    param_names: list(str)
        List of parameter names for plotting.
    bins: int, default: 25
        Bins for histogram plot
    dpi: int, default: 300
        Dots per inch (dpi) for plot
    figsize: tuple(int, int), default: (24, 12)
        Figure size
    interval: float, default: 0.99
        Interval to plot
    show: bool, default: True
        Controls whether the plot shall be printed
    font_size: int, default:12
        Font size

    """

    # Plot settings
    plt.rcParams['font.size'] = font_size
    N = int(theta_test.shape[0])

    # Determine n_subplots dynamically
    n_row = int(np.ceil(len(param_names) / 6))
    n_col = int(np.ceil(len(param_names) / n_row))

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat

    # Compute ranks (using broadcasting)    
    ranks = np.sum(theta_samples < theta_test, axis=0)
    
    # Compute interval
    endpoints = binom.interval(interval, N, 1 / (bins+1))

    # Plot histograms
    for j in range(len(param_names)):
        
        # Add interval
        axarr[j].axhspan(endpoints[0], endpoints[1], facecolor='gray', alpha=0.3)
        axarr[j].axhline(np.mean(endpoints), color='gray', zorder=0, alpha=0.5)
        
        sns.histplot(ranks[:, j], kde=False, ax=axarr[j], color='#a34f4f', bins=bins, alpha=0.95)
        
        axarr[j].set_title(param_names[j])
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)
        if j == 0:
            axarr[j].set_xlabel('Rank statistic')
        axarr[j].get_yaxis().set_ticks([])
        axarr[j].set_ylabel('')
    
    f.tight_layout()
    
    # Show, if specified
    if show:
        plt.show()
    return f


def plot_confusion_matrix(m_true, m_pred, model_names, normalize=False, 
                          cmap=plt.cm.Blues, figsize=(14, 8), annotate=True, show=True):
    """A function to print and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.

    Parameters
    ----------
    m_true: np.array
        Array of true model indices
    m_pred: np.array
        Array of predicted model indices
    model_names: list(str)
        List of model names for plotting
    normalize: bool, default: False
        Controls whether normalization shall be applied
    cmap: matplotlib.pyplot.cm.*, default: plt.cm.Blues
        Colormap
    figsize: tuple(int, int), default: (14, 8)
        Figure size
    annotate: bool, default: True
        Controls if the plot shall be annotated
    show: bool, default: True
        Controls if the plot shall be printed

    """

    # Take argmax of test
    m_true = np.argmax(m_true.numpy(), axis=1).astype(np.int32)


    # Compute confusion matrix
    cm = confusion_matrix(m_true, m_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=model_names, yticklabels=model_names,
           ylabel='True Model',
           xlabel='Predicted Model')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.

    if annotate:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_expected_calibration_error(m_true, m_pred, n_bins=15):
    """Estimates the calibration error of a model comparison neural network.

    Important
    ---------
    Make sure that ``m_true`` are **one-hot encoded** classes!

    Parameters
    ----------
    m_true: np.array or list
        True model indices
    m_pred: np.array or list
        Predicted model indices
    n_bins: int, default: 15
        Number of bins for plot
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

    return np.array(cal_errs), np.array(probs)


def plot_calibration_curves(cal_probs, cal_errs, model_names, font_size=12, figsize=(12, 4)):
    """Plots the calibration curves for a model comparison problem.

    Parameters
    ----------
    cal_probs: np.array
        Currently not used
    cal_errs: np.array or list
        Array of calibration curve data
    model_names: list(str)
        List of model names for plotting
    font_size: int, default: 12
        Font size
    figsize: tuple(int, int), default: (12, 4)
        Figure size for plot layout

    """

    plt.rcParams['font.size'] = 12
    n_models = len(model_names)

    # Determine figure layout
    if n_models >= 6:
        n_col = int(np.sqrt(n_models))
        n_row = int(np.sqrt(n_models)) + 1
    else:
        n_col = n_models
        n_row = 1

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat

    # Loop through models
    for i, ax in enumerate(axarr.flat):

        # Plot calibration curve
        ax.plot(cal_errs[i, 0, :], cal_errs[i, 1, :])

        # Plot AB line
        ax.plot(ax.get_xlim(), ax.get_xlim(), '--', color='black')

        # Tweak plot
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Confidence')
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.text(0.1, 0.9, r'$\widehat{ECE}$ = {0:.3f}'.format(cal_errs[i]),
                        horizontalalignment='left',
                        verticalalignment='center',
                        transform=ax.transAxes,
                        size=12)

        # Set title
        ax.set_title(model_names[i])
    f.tight_layout()
    return f