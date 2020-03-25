import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix


def plot_true_est_scatter(model, X_test, theta_test, n_samples, param_names, 
                          figsize=(20, 4), theta_approx_means=None, show=True, filename=None, font_size=12):
    """Plots a scatter plot with abline of the estimated posterior means vs true values."""


    # Plot settings
    plt.rcParams['font.size'] = font_size

    # Convert true parameters to numpy
    theta_test = theta_test.numpy()

    # Determine figure layout
    if len(param_names) >= 6:
        n_col = int(np.ceil(len(param_names) / 2))
        n_row = 2
    else:
        n_col = int(len(param_names))
        n_row = 1

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat

    # Initialize posterior means matrix, if nose specified
    if theta_approx_means is None:
        theta_approx_means = model.sample(X_test, n_samples, to_numpy=True).mean(axis=0)
        
    # --- Plot true vs estimated posterior means on a single row --- #
    for j in range(len(param_names)):
        
        # Plot analytic vs estimated
        axarr[j].scatter(theta_approx_means[:, j], theta_test[:, j], color='black', alpha=0.4)
        
        # get axis limits and set equal x and y limits
        lower_lim = min(axarr[j].get_xlim()[0], axarr[j].get_ylim()[0])
        upper_lim = max(axarr[j].get_xlim()[1], axarr[j].get_ylim()[1])
        axarr[j].set_xlim((lower_lim, upper_lim))
        axarr[j].set_ylim((lower_lim, upper_lim))
        axarr[j].plot(axarr[j].get_xlim(), axarr[j].get_xlim(), '--', color='black')
        
        # Compute NRMSE
        rmse = np.sqrt(np.mean( (theta_approx_means[:, j] - theta_test[:, j])**2 ))
        nrmse = rmse / (theta_test[:, j].max() - theta_test[:, j].min())
        axarr[j].text(0.1, 0.9, 'NRMSE={:.3f}'.format(nrmse),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes,
                     size=10)
        
        # Compute R2
        r2 = r2_score(theta_test[:, j], theta_approx_means[:, j])
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
    
    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_{}n_scatter.png".format(filename, X_test.shape[1]), dpi=600)


def plot_losses(losses, figsize=(15, 5), show=True):
    """
    Plots ML loss and decay for a given training session.
    ----------

    Arguments:
    losses  : dict -- a dictionary with keys 'ml_loss' and 'decay' containing the portions of the loss.
    figsize : tuple -- the size of the figure to create 
    show    : bool -- a flag indicating whether to call plt.show() or not
    """
    
    f, axarr = plt.subplots(1, 2, figsize=figsize)
    axarr[0].plot(losses['ml_loss'])
    axarr[1].plot(losses['decay'])
    axarr[0].set_title('ML Loss')
    axarr[1].set_title('Decay')
    f.suptitle('Course of Loss')

    if show:
        plt.show()


def plot_metrics(metrics, ns, param_names, figsize=(12, 4), show=True, 
                 xlabel=r'$n$', filename=None, font_size=12):
    """
    Plots the nrmse and r2 for all parameters and all time points.
    """

    # Plot settings
    plt.rcParams['font.size'] = font_size
    
    # Initialize figure
    f, axarr = plt.subplots(1, 2, figsize=figsize)

    for i, metric in enumerate(['nrmse', 'r2']):
        for p in param_names:
            sns.lineplot(ns, metrics[metric][p], label=p, markers=True, dashes=False, ax=axarr[i])
            
        if metric == 'nrmse':
            axarr[i].set_ylabel('NRMSE')
        elif metric == 'r2':
            axarr[i].set_ylabel(r'$R^{2}$')
        axarr[i].set_xlabel(xlabel)
        
        axarr[i].spines['right'].set_visible(False)
        axarr[i].spines['top'].set_visible(False)
        axarr[i].legend(fontsize=12)
    
    f.tight_layout()
        
    if show:
        plt.show()
    
    if filename is not None:
        f.savefig("figures/{}_metrics.png".format(filename), dpi=600, bbox_inches='tight')


def plot_metrics_params(model, X_test, theta_test, n_samples, n_chunks=None, show=True, font_size=12):
    """Plots R2 and NRMSE side by side for all parameters over a test set."""
    
    # Plot initialization
    plt.rcParams['font.size'] = font_size
    f, axarr = plt.subplots(1, 2, figsize=(10, 4))

    # Convert true parameters to numpy
    theta_test = theta_test.numpy()
    
    # Compute posterior means (may need to do this in chunks, if parameter space is too big
    # in order to avoid MemoryError)
    
    if n_chunks is None:
        theta_approx_means = model.sample(X_test, n_samples, to_numpy=True).mean(axis=0)
    else:
        theta_approx_means = np.concatenate(
                        [model.sample(X_test, n_samples // n_chunks, to_numpy=True)
                         for _ in range(n_chunks)], axis=0).mean(axis=0)

    # Compute NRMSE
    rmse = np.sqrt( np.mean( (theta_approx_means - theta_test)**2, axis=0) )
    nrmse = rmse / (theta_test.max(axis=0) - theta_test.min(axis=0))

    # Compute R2
    r2 = r2_score(theta_test, theta_approx_means, multioutput='raw_values')
    
    # Plot NRMSE
    sns.lineplot(np.arange(theta_test.shape[1]) + 1, nrmse, 
                 markers=True, dashes=False, ax=axarr[0])
    # Plot R2
    sns.lineplot(np.arange(theta_test.shape[1]) + 1, r2, 
                 markers=True, dashes=False, ax=axarr[1])
    
    # Tweak plot of NRMSE
    axarr[0].set_xlabel('Parameter #')
    axarr[0].set_ylabel('NRMSE')
    axarr[0].set_title('Test NRMSE')
    axarr[0].spines['right'].set_visible(False)
    axarr[0].spines['top'].set_visible(False)
    
    # Tweak plot of R2
    axarr[1].set_xlabel('Parameter #')
    axarr[1].set_ylabel('$R^2$')
    axarr[1].set_title('Test $R^2$')
    axarr[1].spines['right'].set_visible(False)
    axarr[1].spines['top'].set_visible(False)
    
    f.tight_layout()
        
    if show:
        plt.show()


def plot_variance(variances, ns, param_names, figsize=(12, 4), show=True, 
                  xlabel=r'$n$', filename=None, tight=True, std=False, font_size=12):
    """
    Plots posterior variances of parameters as a function of the number of time points.
    """

    # Plot settings
    plt.rcParams['font.size'] = font_size

    # Initialize figure
    # Determine figure layout
    if len(param_names) >= 6:
        n_col = int(np.ceil(len(param_names) / 2))
        n_row = 2
    else:
        n_col = int(len(param_names))
        n_row = 1

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat


    for i, p in enumerate(param_names):
        
        if std:
            sns.lineplot(ns, np.sqrt(variances[p]), markers=True, dashes=False, ax=axarr[i])
        else:
            sns.lineplot(ns, variances[p], markers=True, dashes=False, ax=axarr[i])
        
        if i == 0:
            axarr[i].set_ylabel('Posterior variance')
            axarr[i].set_xlabel(xlabel)
        axarr[i].set_title(p)
        axarr[i].spines['right'].set_visible(False)
        axarr[i].spines['top'].set_visible(False)

    if tight:
        f.tight_layout()
        
    if show:
        plt.show()
    
    if filename is not None:
        f.savefig("figures/{}_variance.png".format(filename), dpi=600, bbox_inches='tight')


def plot_true_est_posterior(model, n_samples, param_names, n_test=None, data_generator=None, 
                            X_test=None, theta_test=None, figsize=(15, 20), tight=True, 
                            show=True, filename=None, font_size=12):
    """
    Plots approximate posteriors.
    """

    # Plot settings
    plt.rcParams['font.size'] = font_size
    
    if data_generator is None and n_test is None:
        n_test = int(X_test.shape[0])
    elif X_test is None and theta_test is None:
        X_test, theta_test = data_generator(n_test)
    else:
        raise ValueError('Either data_generator and n_test or X_test and y_test should be provided')

    # Convert theta to numpy
    theta_test = theta_test.numpy()

    # Initialize f
    f, axarr = plt.subplots(n_test, len(param_names), figsize=figsize)

    theta_samples = model.sample(X_test, n_samples, to_numpy=True)
    theta_samples_means = theta_samples.mean(axis=0)
    
    # For each row 
    for i in range(n_test):
        
        for j in range(len(param_names)):
            
            
            # Plot approximate posterior
            sns.distplot(theta_samples[:, i, j], kde=False, hist=True, ax=axarr[i, j], 
                            label='Estimated posterior', color='#5c92e8')
            
            # Plot lines for approximate mean, analytic mean and true data-generating value
            axarr[i, j].axvline(theta_samples_means[i, j], color='#5c92e8', label='Estimated mean')
            axarr[i, j].axvline(theta_test[i, j], color='#e55e5e', label='True')
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
            axarr[i, j].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axarr[i, j].get_yaxis().set_ticks([])
            
            
            # Set title of first row
            if i == 0:
                axarr[i, j].set_title(param_names[j])       
            
            if i == 0 and j == 0:
                f.legend(loc='lower center', bbox_to_anchor=(0.5, -0.03), shadow=True, ncol=3, fontsize=10, borderaxespad=1)
                #axarr[i, j].legend(fontsize=10)
                
    if tight:
        f.tight_layout()
    f.subplots_adjust(bottom=0.12)
    # Show, if specified
    if show:
        plt.show()
    
    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_{}n_density.png".format(filename, X_test.shape[1]), dpi=600, bbox_inches='tight')


def plot_true_est_posterior_samples(theta_samples, theta_test, param_names, figsize=(15, 20), 
                                    tight=True, show=True, filename=None, font_size=12):
    """
    Plots approximate posteriors.
    """

    # Plot settings
    plt.rcParams['font.size'] = font_size
    
    # Convert theta to numpy
    theta_test = theta_test.numpy()
    n_test = theta_test.shape[0]

    # Initialize f
    f, axarr = plt.subplots(n_test, len(param_names), figsize=figsize)
    axarr = np.atleast_2d(axarr)

    theta_samples_means  = np.mean(theta_samples, axis=0, keepdims=1)

    # For each row 
    for i in range(n_test):
        for j in range(len(param_names)):
            
            
            # Plot approximate posterior
            if len(theta_samples.shape) == 3:
                theta_samples_p = theta_samples[:, i, j]
            else:
                theta_samples_p = theta_samples[:, j]

            sns.distplot(theta_samples_p, kde=True, hist=True, ax=axarr[i, j], 
                            label='Estimated posterior', color='#5c92e8')
            
            # Plot lines for approximate mean, analytic mean and true data-generating value
            axarr[i, j].axvline(theta_samples_means[i, j], color='#5c92e8', label='Estimated mean')
            axarr[i, j].axvline(theta_test[i, j], color='#e55e5e', label='True')
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
            axarr[i, j].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axarr[i, j].get_yaxis().set_ticks([])
            
            
            # Set title of first row
            if i == 0:
                axarr[i, j].set_title(param_names[j])       
            
            if i == 0 and j == 0:
                f.legend(loc='lower center', bbox_to_anchor=(0.5, -0.03), shadow=True, ncol=3, fontsize=10, borderaxespad=1)
                #axarr[i, j].legend(fontsize=10)
                
    if tight:
        f.tight_layout()
    f.subplots_adjust(bottom=0.12)
    # Show, if specified
    if show:
        plt.show()
    
    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_{}n_density.png".format(filename, X_test.shape[1]), dpi=600, bbox_inches='tight')


def plot_sbc(theta_samples, theta_test, param_names, bins=None,
            figsize=(15, 5), show=True, filename=None, font_size=12):
    """
    Plots the simulation-based posterior checking histograms as advocated by Talts et al. (2018).
    """

    # Plot settings
    plt.rcParams['font.size'] = font_size
    
    # Prepare figure
    if len(param_names) >= 6:
        n_col = int(np.ceil(len(param_names) / 2))
        n_row = 2
    else:
        n_col = int(len(param_names))
        n_row = 1
    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat

    # Convert theta test to numpy
    theta_test = theta_test.numpy()

    # Sample from approximate posterior
    

    # Compute ranks (using broadcasting)    
    ranks = np.sum(theta_samples < theta_test, axis=0)

    # Plot histograms
    for j in range(len(param_names)):
        sns.distplot(ranks[:, j], kde=False, ax=axarr[j], rug=True, hist_kws=dict(edgecolor="k", linewidth=1), bins=bins)
        axarr[j].set_title(param_names[j])
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)
        if j == 0:
            axarr[j].set_xlabel('Rank statistic')
        axarr[j].get_yaxis().set_ticks([])

    f.tight_layout()
    
    # Show, if specified
    if show:
        plt.show()
    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_{}n_sbc.png".format(filename, X_test.shape[1]), dpi=600)


def plot_model_samples(model, x_test, m_test, model_names, n_samples=5000, 
                       figsize=(15, 9), fontsize=9, show=True, filename=None):
    """
    Plots model probabilities. Best viewed when x_test and m_test contain 36 test datasets.
    """
    
    n_test = int(x_test.shape[0])
    m_samples = model.sample(x_test, n_samples=n_samples)
    m_test = np.argmax(m_test.numpy(), axis=1)

    f, axarr = plt.subplots(int(np.sqrt(n_test)), int(np.sqrt(n_test)), figsize=figsize)
    
    # For each test dataset
    for i, ax in enumerate(axarr.flat):
        
        # For each model
        for m in range(model.M):
            # Plot dirichlet samples
            sns.distplot(m_samples[:, i, m], ax=ax, label=model_names[m])
        
        if i == 0:
            ax.legend(fontsize=fontsize)
            
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('True: {}'.format(model_names[m_test[i]]))

    f.tight_layout()

    # Show, if specified
    if show:
        plt.show()
    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_dirichlet.png".format(filename), dpi=600)


def plot_confusion_matrix(m_hat, m_test, model_names, normalize=False, cmap=plt.cm.Blues, figsize=(14, 8), annotate=True, show=True):
    """
    A function to print and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    """

    # Take argmax of test
    m_test = np.argmax(m_test.numpy(), axis=1).astype(np.int32)


    # Compute confusion matrix
    cm = confusion_matrix(m_test, m_hat)

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


