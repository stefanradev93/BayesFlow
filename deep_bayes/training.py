"""
This module implements the basic functions for training deep learning models
for parameter estimation and model selection.
"""


__version__ = '0.1'
__author__ = 'Stefan Radev'

from collections.abc import Iterable
import warnings
import tensorflow as tf
import numpy as np
np.seterr(all = 'raise')
from deep_bayes.losses import maximum_mean_discrepancy, heteroscedastic_loglik, kullback_leibler_gaussian

from .utils import clip_gradients, apply_gradients


def train_online(model, optimizer, data_gen, loss_fun, iterations, batch_size, p_bar=None,
                 clip_value=5., clip_method='global_norm', global_step=None, n_smooth=100, method='flow'):
    """
    Performs a number of training iterations with a given tensorflow model and optimizer.

    ----------

    Arguments:
    model           : tf.keras.Model -- a neural network model implementing a __call__() method
    optimizer       : tf.train.Optimizer -- the optimizer used for backprop
    data_gen        : callable -- a function providing batches of data
    loss_fun        : callable -- a function computing the loss given model outputs
    iterations      : int -- the number of training loops to perform
    batch_size      : int -- the batch_size used for training
    ----------

    Keyword Arguments:
    p_bar           : ProgressBar or None -- an instance for tracking the training progress
    clip_value      : float       -- the value used for clipping the gradients
    clip_method     : str         -- the method used for clipping (default 'global_norm')
    global_step     : tf.Variavle -- a scalar tensor tracking the number of steps and used for learning rate decay  
    ----------

    Returns:
    losses : a dictionary with regularization and loss evaluations at each training iteration
    """
    
    # Prepare a dict for storing losses
    losses = {
        'loss': [],
        'regularization': []
    }

    # Run training loop
    for it in range(1, iterations+1):


        # Generate inputs for the network
        try:
            batch = data_gen(batch_size)
        except RuntimeError:
            print('Runtime warning, skipping batch...')
            p_bar.update(1)
            continue
        except FloatingPointError:
            print('Floating point error, skipping batch...')
            p_bar.update(1)
            continue

        with tf.GradientTape() as tape:
            if method == 'flow':
                inputs = (batch['theta'], batch['x'])
            else:
                inputs = (batch['x'],)

            # Forward pass 
            outputs = model(*inputs)
        
            # Loss computation and backward pass
            if method == 'flow':
                loss_args = (outputs['z'], outputs['log_det_J'])
            else:
                loss_args = (batch['m'], outputs['alpha'], outputs['alpha0'], outputs['m_probs'])
            loss = loss_fun(*loss_args)
            # Compute loss + regularization, if any
            w_decay = tf.add_n(model.losses) if model.losses else 0.
            total_loss = loss + w_decay

        # One step backprop
        gradients = tape.gradient(total_loss, model.trainable_variables)
        if clip_value is not None:
            gradients = clip_gradients(gradients, clip_value, clip_method)
        apply_gradients(optimizer, gradients, model.trainable_variables, global_step)  

        # Store losses
        losses['regularization'].append(w_decay)
        losses['loss'].append(loss)
        running_loss = loss if it < n_smooth else np.mean(losses['loss'][-n_smooth:])

        # Update progress bar
        if p_bar is not None:
            p_bar.set_postfix_str("Iteration: {0},Loss: {1:.3f},Running Loss: {2:.3f},Regularization: {3:.3f}"
            .format(it, loss, running_loss, w_decay))
            p_bar.update(1)
    return losses


def train_online_vae(model, optimizer, data_gen, iterations, batch_size, p_bar=None, regularization='kl',
                     regularization_weight=1.0, clip_value=5., clip_method='global_norm', 
                     global_step=None, n_smooth=100):
    """
    Performs a number of training iterations with an information maximizing VAE.

    ----------

    Arguments:
    model           : tf.keras.Model -- a neural network model implementing a __call__() method
    optimizer       : tf.train.Optimizer -- the optimizer used for backprop
    data_gen        : callable -- a function providing batches of data
    iterations      : int -- the number of training loops to perform
    batch_size      : int -- the batch_size used for training
    ----------

    Keyword Arguments:
    p_bar           : ProgressBar or None -- an instance for tracking the training progress
    clip_value      : float       -- the value used for clipping the gradients
    clip_method     : str         -- the method used for clipping (default 'global_norm')
    global_step     : tf.Variavle -- a scalar tensor tracking the number of steps and used for learning rate decay  
    ----------

    Returns:
    losses : a dictionary with regularization and loss evaluations at each training iteration
    """
    
    # Prepare a dict for storing losses
    losses = {
        'cent': [],
        'regularization': [],
        'total': []
    }

    # Run training loop
    for it in range(1, iterations+1):

        with tf.GradientTape() as tape:

            # Generate inputs for the network
            batch = data_gen(batch_size)

            # Forward pass 
            out = model(batch['x'], batch['m'])

            # Compute losses
            cent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch['m'], logits=out['m_logits']))

            if regularization == 'MMD':
                # Sample from unit Gaussian
                z_true = tf.random_normal(shape=out['z_samples'].shape)
                reg = maximum_mean_discrepancy(out['z_samples'], z_true, weight=regularization_weight)
            elif regularization == 'KL':
                reg = kullback_leibler_gaussian(out['z_mean'], out['z_logvar'], beta=regularization_weight)
            total_loss = cent + reg

        # One step backprop
        gradients = tape.gradient(total_loss, model.trainable_variables)
        if clip_value is not None:
            gradients = clip_gradients(gradients, clip_value, clip_method)
        apply_gradients(optimizer, gradients, model.trainable_variables, global_step)  

        # Store losses
        losses['regularization'].append(reg)
        losses['cent'].append(cent)
        losses['total'].append(total_loss)
        
        running_loss = total_loss if it < n_smooth else np.mean(losses['total'][-n_smooth:])

        # Update progress bar
        if p_bar is not None:
            p_bar.set_postfix_str("Iteration: {0},Loss: {1:.3f},Running Loss: {2:.3f},Cross-Entropy: {3:.3f},{4}: {5:.3f}"
            .format(it, total_loss, running_loss, cent.numpy(), regularization, reg.numpy()))
            p_bar.update(1)
    return losses


def train_online_dropout(model, optimizer, data_gen, iterations, batch_size, p_bar=None,
                         clip_value=5., clip_method='global_norm', global_step=None, n_smooth=100):
    """
    Performs a number of training iterations with heteroscedastic dropout.
    ----------

    Arguments:
    model           : tf.keras.Model -- a neural network model implementing a __call__() method
    optimizer       : tf.train.Optimizer -- the optimizer used for backprop
    data_gen        : callable -- a function providing batches of data
    iterations      : int -- the number of training loops to perform
    batch_size      : int -- the batch_size used for training
    ----------

    Keyword Arguments:
    p_bar           : ProgressBar or None -- an instance for tracking the training progress
    clip_value      : float       -- the value used for clipping the gradients
    clip_method     : str         -- the method used for clipping (default 'global_norm')
    global_step     : tf.Variavle -- a scalar tensor tracking the number of steps and used for learning rate decay  
    ----------

    Returns:
    losses : a dictionary with regularization and loss evaluations at each training iteration
    """
    
    # Prepare a dict for storing losses
    losses = {
        'total': [],
    }

    # Run training loop
    for it in range(1, iterations+1):

        with tf.GradientTape() as tape:

            # Generate inputs for the network
            batch = data_gen(batch_size)

            # Forward pass 
            out = model(batch['x'])
        
            # Compute loss
            total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch['m'], logits=out['m_logits']))

        # One step backprop
        gradients = tape.gradient(total_loss, model.trainable_variables)
        if clip_value is not None:
            gradients = clip_gradients(gradients, clip_value, clip_method)
        apply_gradients(optimizer, gradients, model.trainable_variables, global_step)  

        # Store losses
        losses['total'].append(total_loss)
        
        running_loss = total_loss if it < n_smooth else np.mean(losses['total'][-n_smooth:])

        # Update progress bar
        if p_bar is not None:
            p_bar.set_postfix_str("Iteration: {0},Loss: {1:.3f},Running Loss: {2:.3f}"
            .format(it, total_loss, running_loss))
            p_bar.update(1)
    return losses


def train_online_softmax(model, optimizer, data_gen, iterations, batch_size, p_bar=None,
                         clip_value=5., clip_method='global_norm', global_step=None, n_smooth=100):
    """
    Performs a number of training iterations with a softmax network.

    ----------

    Arguments:
    model           : tf.keras.Model -- a neural network model implementing a __call__() method
    optimizer       : tf.train.Optimizer -- the optimizer used for backprop
    data_gen        : callable -- a function providing batches of data
    iterations      : int -- the number of training loops to perform
    batch_size      : int -- the batch_size used for training
    ----------

    Keyword Arguments:
    p_bar           : ProgressBar or None -- an instance for tracking the training progress
    clip_value      : float       -- the value used for clipping the gradients
    clip_method     : str         -- the method used for clipping (default 'global_norm')
    global_step     : tf.Variavle -- a scalar tensor tracking the number of steps and used for learning rate decay  
    ----------

    Returns:
    losses : a dictionary with regularization and loss evaluations at each training iteration
    """
    
    # Prepare a dict for storing losses
    losses = {
        'total': [],
    }

    # Run training loop
    for it in range(1, iterations+1):

        with tf.GradientTape() as tape:

            # Generate inputs for the network
            batch = data_gen(batch_size)

            # Forward pass 
            out = model(batch['x'])
        
            # Compute losses
            total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out['m_logits'], labels=batch['m']))

        # One step backprop
        gradients = tape.gradient(total_loss, model.trainable_variables)
        if clip_value is not None:
            gradients = clip_gradients(gradients, clip_value, clip_method)
        apply_gradients(optimizer, gradients, model.trainable_variables, global_step)  

        # Store losses
        losses['total'].append(total_loss)
        
        running_loss = total_loss if it < n_smooth else np.mean(losses['total'][-n_smooth:])

        # Update progress bar
        if p_bar is not None:
            p_bar.set_postfix_str("Iteration: {0},Loss: {1:.3f},Running Loss: {2:.3f}"
            .format(it, total_loss, running_loss))
            p_bar.update(1)
    return losses


def train_offline(model, optimizer, dataset, loss_fun, batch_size, p_bar=None, clip_value=5., 
                  clip_method='global_norm', global_step=None, method='flow'):
    """
    Loops throuhg a dataset  #TODO 
    ----------

    Arguments:
    model           : tf.keras.Model -- a neural network model implementing a __call__() method
    optimizer       : tf.train.Optimizer -- the optimizer used for backprop
    data_generator  : callable -- a function providing batches of data
    loss_fun        : callable -- a function computing the loss given model outputs
    batch_size      : int -- the batch_size used for training
    ----------

    Keyword Arguments:
    p_bar           : ProgressBar or None -- an instance for tracking the training progress
    clip_value      : float       -- the value used for clipping the gradients
    clip_method     : str         -- the method used for clipping (default 'global_norm')
    global_step     : tf.Variavle -- a scalar tensor tracking the number of steps and used for learning rate decay  
    ----------

    Returns:
    losses : a dictionary with regularization and loss evaluations at each training iteration
    """
    
    # Prepare a dictionary to track losses
    losses = {
        'loss': [],
        'regularization': []
    }

    # Loop through dataset
    for bi, batch in enumerate(dataset):

        with tf.GradientTape() as tape:

            if method == 'flow':
                inputs = (batch[0], batch[1])
            else:
                inputs = (batch[0], )

            # Forward pass 
            outputs = model(*inputs)
        
            # Loss computation and backward pass
            if method == 'flow':
                loss_args = (outputs['z'], outputs['log_det_J'])
            else:
                loss_args = (batch[1], outputs['alpha'], outputs['alpha0'], outputs['m_probs'])

            loss = loss_fun(*loss_args)
            # Compute loss + regularization, if any
            w_decay = tf.add_n(model.losses) if model.losses else 0.
            total_loss = loss + w_decay

        # One step backprop
        gradients = tape.gradient(total_loss, model.trainable_variables)
        if clip_value is not None:
            gradients = clip_gradients(gradients, clip_value, clip_method)
        apply_gradients(optimizer, gradients, model.trainable_variables, global_step)  

        # Store losses
        losses['regularization'].append(w_decay)
        losses['loss'].append(loss)

        # Update progress bar
        if p_bar is not None:
            p_bar.set_postfix_str("Batch: {0},Loss: {1:.3f},Regularization: {2:.3f}"
            .format(bi, loss, w_decay))
            p_bar.update(1)
    return losses
