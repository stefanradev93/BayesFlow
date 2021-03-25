import numpy as np
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.train import CheckpointManager, Checkpoint
from tensorflow.keras.utils import to_categorical

class MetaTrainer:
    
    def __init__(self, network, generative_model, loss, summary_stats=None, optimizer=None,
                learning_rate=0.0005, checkpoint_path=None, max_to_keep=5, clip_method='global_norm', clip_value=None):
        """
        Creates a trainer instance for performing single-model forward inference and training an
        amortized neural estimator for parameter estimation (BayesFlow). If a checkpoint_path is provided, the
        network's weights will be stored after each training epoch. If the folder contains a 
        checkpoint, the trainer will try to load the weights and continue training with a pre-trained net.
        ----------
        
        Arguments:
        network     : bayesflow.Amortizer instance -- the neural architecture to be optimized
        generative_model: callable -- a function or an object with n_sim and n_obs mandatory arguments 
                          returning randomly sampled parameter vectors and datasets from a process model
        loss        : callable with three arguments: (network, m_indices, x) -- the loss function
        ----------
        
        Keyword arguments:
        summary_stats   : callable -- optional summary statistics function
        optimizer       : None or tf.keras.optimizer.Optimizer -- default Adam optimizer (equiv. to None) or a custom one
        learning_rate   : float -- the learning rate used for the optimizer
        checkpoint_path : string -- optional folder name for storing the trained network
        max_to_keep     : int -- optional number of checkpoints to keep
        clip_method     : string in ('norm', 'value', 'global_norm') -- optional gradient clipping method
        clip_value      : float -- the value used for gradient clipping when clip_method is set to 'value' or 'norm'
        """
        
        # Basic attributes
        self.network = network
        self.generative_model = generative_model
        self.loss = loss
        self.summary_stats = summary_stats
        self.clip_method = clip_method
        self.clip_value = clip_value
        self.n_obs = None
        
        # Optimizer settings
        if optimizer is None:
            if tf.__version__.startswith('1'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                self.optimizer = Adam(learning_rate)
        else:
            self.optimizer = optimizer(learning_rate)

        # Checkpoint settings
        if checkpoint_path is not None:
            self.checkpoint = Checkpoint(optimizer=self.optimizer, model=self.network)
            self.manager = CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=max_to_keep)
            self.checkpoint.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Networks loaded from {}".format(self.manager.latest_checkpoint))
            else:
                print("Initializing networks from scratch.")
        else:
            self.checkpoint = None
            self.manager = None
        self.checkpoint_path = checkpoint_path

    def train_online(self, epochs, iterations_per_epoch, batch_size, n_obs, **kwargs):
        """
        Trains the inference network(s) via online learning. Additional keyword arguments
        are passed to the simulators.
        ----------
        
        Arguments:
        epochs               : int -- number of epochs (and number of times a checkpoint is stored)
        iterations_per_epoch : int -- number of batch simulations to perform per epoch
        batch_size           : int -- number of simulations to perform at each backprop step
        n_obs                : int or callable -- if int, then treated as a fixed number of observations, if callable, then
                               treated as a function for sampling N, i.e., N ~ p(N)
        ----------

        Returns:
        losses : dict (ep_num : list_of_losses) -- a dictionary storing the losses across epochs and iterations
        """
        
        losses = dict()
        for ep in range(1, epochs+1):
            losses[ep] = []
            with tqdm(total=iterations_per_epoch, desc='Training epoch {}'.format(ep)) as p_bar:
                for it in range(1, iterations_per_epoch+1):

                    # Determine n_obs and generate data on-the-fly
                    if type(n_obs) is int:
                        n_obs_it = n_obs
                    else:
                        n_obs_it = n_obs()
                    model_indices, params, sim_data = self._forward_inference(batch_size, n_obs_it, **kwargs)

                    # One step backprop
                    loss = self._train_step(model_indices, params, sim_data)

                    # Store loss into dictionary
                    losses[ep].append(loss)
                    
                    # Update progress bar
                    p_bar.set_postfix_str("Epoch {0},Iteration {1},Loss: {2:.3f},Running Loss: {3:.3f}"
                    .format(ep, it, loss, np.mean(losses[ep])))
                    p_bar.update(1)

            # Store after each epoch, if specified
            if self.manager is not None:
                self.manager.save()
        return losses

    def train_offline(self, epochs, batch_size, model_indices, params, sim_data):
        """
        Trains the inference network(s) via offline learning. Assume params and data have already 
        been simulated (i.e., forward inference). 
        ----------
        
        Arguments:
        epochs           : int -- number of epochs (and number of times a checkpoint is stored)
        batch_size       : int -- number of simulations to perform at each backprop step
        model_indices    : np.array of shape (n_sim, ) or (n_sim, n_models) -- the true model indices
        params           : np.array of shape (n_sim, n_params) -- the true data-generating parameters
        sim_data         : np.array of shape (n_sim, n_obs, data_dim) -- the simulated data sets from each model
        ----------

        Returns:
        losses : dict (ep_num : list_of_losses) -- a dictionary storing the losses across epochs and iterations
        """

        # Convert to a data set
        n_sim = int(sim_data.shape[0])

        # Compute summary statistics, if provided
        if self.summary_stats is not None:
            print('Computing hand-crafted summary statistics...')
            sim_data = self.summary_stats(sim_data)

        print('Converting {} simulations to a TensorFlow data set...'.format(n_sim))
        data_set = tf.data.Dataset \
                    .from_tensor_slices((model_indices, params, sim_data)) \
                    .shuffle(n_sim) \
                    .batch(batch_size)

        losses = dict()
        for ep in range(1, epochs+1):
            losses[ep] = []
            with tqdm(total=int(np.ceil(n_sim / batch_size)), desc='Training epoch {}'.format(ep)) as p_bar:
                # Loop through dataset
                for bi, batch in enumerate(data_set):

                    # Extract params from batch
                    model_indices_b, params_b, sim_data_b = batch[0], batch[1], batch[2]

                    # One step backprop
                    loss = self._train_step(model_indices_b, params_b, sim_data_b)

                    # Store loss and update progress bar
                    losses[ep].append(loss)
                    p_bar.set_postfix_str("Epoch {0},Batch {1},Loss: {2:.3f},Running Loss: {3:.3f}"
                    .format(ep, bi+1, loss, np.mean(losses[ep])))
                    p_bar.update(1)
                
            # Store after each epoch, if specified
            if self.manager is not None:
                self.manager.save()
        return losses

    def train_rounds(self, epochs, rounds, sim_per_round, batch_size, n_obs, **kwargs):
        """
        Trains the inference network(s) via round-based learning. Additional arguments are
        passed to the simulator.
        ----------
        
        Arguments:
        epochs         : int -- number of epochs (and number of times a checkpoint is stored)
        rounds         : int -- number of rounds to perform 
        sim_per_round  : int -- number of simulations per round
        batch_size     : int -- number of simulations to perform at each backprop step
        n_obs          : int -- number of observations (fixed) for each data set
        ----------

        Returns:
        losses : nested dict with each (ep_num : list_of_losses) -- a dictionary storing the losses across rounds, 
                 epochs and iterations
        """

        # Make sure n_obs is fixed, otherwise not working 
        assert type(n_obs) is int,\
        'Round-based training currently only works with fixed n_obs. Use online learning for variable n_obs or fix n_obs to an integer value.'

        losses = dict()
        for r in range(1, rounds+1):
            
            # Data generation step
            if r == 1:
                # Simulate initial data
                print('Simulating initial {} data sets...'.format(sim_per_round))
                model_indices, params, sim_data = self._forward_inference(sim_per_round, n_obs, **kwargs)
            else:
                # Simulate further data
                print('Simulating new {} data sets and appending to previous...'.format(sim_per_round))
                print('New total number of simulated data sets: {}'.format(sim_per_round * r))
                model_indices_r, params_r, sim_data_r = self._forward_inference(sim_per_round, n_obs, **kwargs)

                # Add new simulations to previous data
                model_indices = np.concatenate((model_indices, model_indices_r), axis=0)
                params = np.concatenate((params, params_r), axis=0)
                sim_data = np.concatenate((sim_data, sim_data_r), axis=0)

            # Train offline with generated stuff
            losses_r = self.train_offline(epochs, batch_size, model_indices, params, sim_data)
            losses[r] = losses_r

        return losses

    def simulate_and_train_offline(self, n_sim, epochs, batch_size, n_obs, **kwargs):
        """
        Simulates n_sim data sets and then trains the inference network(s) via offline learning. 

        Additional keyword arguments are passed to the simulator.
        ----------
        
        Arguments:
        n_sim          : int -- total number of simulations to perform
        epochs         : int -- number of epochs (and number of times a checkpoint is stored)
        batch_size     : int -- number of simulations to perform at each backprop step
        n_obs          : int -- number of observations for each dataset
        ----------

        Returns:
        losses : dict (ep_num : list_of_losses) -- a dictionary storing the losses across epochs and iterations
        """

        # Make sure n_obs is fixed, otherwise not working, for now
        assert type(n_obs) is int,\
        'Offline training currently only works with fixed n_obs. Use online learning for variable n_obs or fix n_obs to an integer value.'

        # Simulate data
        print('Simulating {} data sets upfront...'.format(n_sim))
        model_indices, params, sim_data = self._forward_inference(n_sim, n_obs, summarize=False, **kwargs)

        # Train offlines
        losses = self.train_offline(epochs, batch_size, model_indices, params, sim_data)
        return losses

    def load_pretrained_network(self):
        """
        Attempts to load a pre-trained network if checkpoint path is provided and a checkpoint manager exists.
        """

        if self.manager is None or self.checkpoint is None:
            return False
        status = self.checkpoint.restore(self.manager.latest_checkpoint)
        return status

    def _forward_inference(self, n_sim, n_obs, summarize=True, **kwargs):
        """
        Performs one step of multi-model forward inference.
        ----------
        
        Arguments:
        n_sim : int -- number of simulation to perform at the given step (i.e., batch size)
        n_obs : int or callable -- if int, then treated as a fixed number of observations, if callable, then
                                   treated as a function for sampling N, i.e., N ~ p(N)
        ----------

        Kyeword arguments:
        summarize : bool -- whether to summarize the data if hand-crafted summaries are given

        Returns:
        params    : np.array (np.float32) of shape (batch_size, param_dim) -- array of sampled parameters
        sim_data  : np.array (np.float32) of shape (batch_size, n_obs, data_dim) -- array of simulated data sets
        """
        
        # Simulate data with n_sims and n_obs
        # Return shape of params is (batch_size, param_dim)
        # Return shape of data is (batch_size, n_obs, data_dim)
        model_indices, params, sim_data = self.generative_model(n_sim, n_obs, **kwargs)

        # Compute hand-crafted summary stats, if given
        if summarize and self.summary_stats is not None:
            # Return shape in this case is (batch_size, n_sum)
            sim_data = self.summary_stats(sim_data)

        return model_indices, params, sim_data

    def _train_step(self, model_indices, params, sim_data):
        """
        Performs one step of backpropagation with the given model indices and data.
        ----------
        
        Arguments:
        model_indices  : np.array (np.float32) of shape (n_sim, n_models) -- the true model indices
        params         : np.array (np.float32) of shape (batch_size, n_params) -- matrix of n_samples x n_params
        sim_data       : np.array (np.float32) of shape (batch_size, n_obs, data_dim) or (batch_size, summary_dim) 
                    -- array of simulated data sets (or summary statistics thereof)      
        ----------

        Returns:
        loss : tf.Tensor of shape (,), i.e., a scalar representing the average loss over the batch of m and x
        """
        
        # Compute loss and store gradients
        with tf.GradientTape() as tape:
            loss = self.loss(self.network, model_indices, params, sim_data)
            
        # One step backprop
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self._apply_gradients(gradients, self.network.trainable_variables)  
        
        return loss.numpy()
        
    def _apply_gradients(self, gradients, tensors):
        """
        Updates each tensor in the 'variables' list via backpropagation. Operation is performed in-place.
        ----------

        Arguments:
        gradients: list of tf.Tensor -- the list of gradients for all neural network parameter
        variables: list of tf.Tensor -- the list of all neural network parameters
        """

        # Optional gradient clipping
        if self.clip_value is not None:
            gradients = clip_gradients(gradients, clip_value=self.clip_value, clip_method=self.clip_method)
        self.optimizer.apply_gradients(zip(gradients, tensors))