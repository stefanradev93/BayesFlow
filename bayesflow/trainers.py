import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tqdm.notebook import tqdm

from bayesflow.buffer import MemoryReplayBuffer
from bayesflow.exceptions import SimulationError, SummaryStatsError
from bayesflow.helpers import clip_gradients
from bayesflow.losses import kl_latent_space, log_loss
from bayesflow.models import MetaGenerativeModel, SimpleGenerativeModel


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
        optimizer       : None or tf.keras.optimizer.Optimizer -- default Adam optimizer (equiv to None) or a custom one
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
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.network)
            self.manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=max_to_keep)
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
        n_obs                : int or callable -- if int, then treated as a fixed number of observations,
                               if callable, then treated as a function for sampling N, i.e., N ~ p(N)
        ----------

        Returns:
        losses : dict (ep_num : list_of_losses) -- a dictionary storing the losses across epochs and iterations
        """

        losses = dict()
        for ep in range(1, epochs + 1):
            losses[ep] = []
            with tqdm(total=iterations_per_epoch, desc='Training epoch {}'.format(ep)) as p_bar:
                for it in range(1, iterations_per_epoch + 1):

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
        for ep in range(1, epochs + 1):
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
                                          .format(ep, bi + 1, loss, np.mean(losses[ep])))
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
        assert type(n_obs) is int, \
            'Round-based training currently only works with fixed n_obs. ' \
            'Use online learning for variable n_obs or fix n_obs to an integer value.'

        losses = dict()
        model_indices = params = sim_data = None
        for r in range(1, rounds + 1):

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
                model_indices = np.concatenate((model_indices, model_indices_r), axis=0) \
                    if model_indices is not None else model_indices
                params = np.concatenate((params, params_r), axis=0) \
                    if params is not None else params
                sim_data = np.concatenate((sim_data, sim_data_r), axis=0) \
                    if sim_data is not None else sim_data

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
        assert type(n_obs) is int, \
            'Offline training currently only works with fixed n_obs. ' \
            'Use online learning for variable n_obs or fix n_obs to an integer value.'

        # Simulate data
        print('Simulating {} data sets upfront...'.format(n_sim))
        model_indices, params, sim_data = self._forward_inference(n_sim, n_obs, summarize=False, **kwargs)

        # Train offline
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

        Keyword arguments:
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


class ModelComparisonTrainer:

    def __init__(self, network, generative_model: MetaGenerativeModel, loss=None, summary_stats=None, optimizer=None,
                 learning_rate=0.0005, checkpoint_path=None, max_to_keep=5, clip_method='global_norm', clip_value=None):
        """
        Creates a trainer instance for performing multi-model forward inference and training an
        amortized neural estimator for model comparison. If a checkpoint_path is provided, the
        network's weights will be stored after each training epoch. If the folder contains a 
        checkpoint, the trainer will try to load the weights and continue training with a pre-trained net.
        ----------
        
        Arguments:
        network     : tf.keras.Model instance -- the neural network to be optimized
        generative_model: callable -- provides (model_indices, params, sim_data) in batch format
        loss        : callable with three mandatory arguments: (network, m_indices, x), if None ('default'), use logloss
        ----------
        
        Keyword arguments:
        summary_stats   : callable -- optional summary statistics function
        learning_rate   : float -- the learning rate used for the optimizer
        checkpoint_path : string -- optional folder name for storing the trained network
        max_to_keep     : int -- optional number of checkpoints to keep
        clip_method     : string in ('norm', 'value', 'global_norm') -- optional gradient clipping method
        clip_value      : float -- the value used for gradient clipping when clip_method is set to 'value' or 'norm'
        """

        # Basic attributes
        self.network = network
        self.generative_model = generative_model
        # Default or custom loss
        if loss is None:
            self.loss = log_loss
        else:
            self.loss = loss
        # Optional hand-crafted summaries
        self.summary_stats = summary_stats
        self.n_models = len(generative_model.generative_models)
        self.clip_method = clip_method
        self.clip_value = clip_value

        # Optimizer settings
        if optimizer is None:
            if tf.__version__.startswith('1'):
                self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            else:
                self.optimizer = Adam(learning_rate)
        else:
            self.optimizer = optimizer(learning_rate)

        # Checkpoint settings
        if checkpoint_path is not None:
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.network)
            self.manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=max_to_keep)
            self.checkpoint.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Networks loaded from {}".format(self.manager.latest_checkpoint))
            else:
                print("Initializing networks from scratch.")
        else:
            self.checkpoint = None
            self.manager = None
        self.checkpoint_path = checkpoint_path

        # TODO - make sure forward inference goes through

    def train_online(self, epochs, iterations_per_epoch, batch_size, n_obs, **kwargs):
        """
        Trains the inference network(s) via online learning. Additional keyword arguments
        will be passed to the simulators.
        ----------
        
        Arguments:
        epochs               : int -- number of epochs (and number of times a checkpoint is stored)
        iterations_per_epoch : int -- number of batch simulations to perform per epoch
        batch_size           : int -- number of simulations to perform at each backprop step
        n_obs                : int or callable -- if int, then treated as a fixed number of observations,
                               if callable, then treated as a function for sampling N, i.e., N ~ p(N)
        ----------

        Returns:
        losses : dict (ep_num : list_of_losses) -- a dictionary storing the losses across epochs and iterations
        """

        losses = dict()
        for ep in range(1, epochs + 1):
            losses[ep] = []
            with tqdm(total=iterations_per_epoch, desc='Training epoch {}'.format(ep)) as p_bar:
                for it in range(iterations_per_epoch):

                    # Determine n_obs for current batch
                    if type(n_obs) is int:
                        n_obs_it = n_obs
                    else:
                        n_obs_it = n_obs()

                    # Generate model indices and data on-the-fly
                    model_indices, sim_data = self._forward_inference(batch_size, n_obs_it, **kwargs)

                    # One step backprop
                    loss = self._train_step(model_indices, sim_data)

                    # Store loss into dict
                    losses[ep].append(loss)

                    # Update progress bar
                    p_bar.set_postfix_str("Epoch {0},Iteration {1},Loss: {2:.3f},Running Loss: {3:.3f}"
                                          .format(ep, it, loss, np.mean(losses[ep])))
                    p_bar.update(1)

            # Store after each epoch, if specified
            if self.manager is not None:
                self.manager.save()
        return losses

    def train_offline(self, epochs, batch_size, model_indices, sim_data, **_kwargs):
        """
        Trains the inference network(s) via offline learning. Additional arguments are passed
        to the train step method.
        ----------
        
        Arguments:
        epochs         : int -- number of epochs (and number of times a checkpoint is stored)
        batch_size     : int -- number of simulations to perform at each backprop step
        model_indices  : np.array of shape (n_sim, ) or (n_sim, n_models) -- the true model indices
        sim_data       : np.array of shape (n_sim, N, data_dim) -- the simulated data sets from each model
        ----------

        Returns:
        losses : dict (ep_num : list_of_losses) -- a dictionary storing the losses across epochs and iterations
        """

        n_sim = int(sim_data.shape[0])

        # If model_indices is 1D, perform one-hot encoding
        if len(model_indices.shape) == 1:
            print('One-hot-encoding model indices...')
            model_indices = to_categorical(model_indices, n_classes=self.n_models)

        # Compute hand-crafted summary statistics, if given
        if self.summary_stats is not None:
            print('Computing hand-crafted summary statistics...')
            sim_data = self.summary_stats(sim_data)

        # Convert to a tensorflow data set. Assumes all data sets have the same shape
        print('Converting {} simulations to a TensorFlow data set...'.format(n_sim))
        data_set = tf.data.Dataset.from_tensor_slices((model_indices, sim_data)).shuffle(n_sim).batch(batch_size)

        losses = dict()
        for ep in range(1, epochs + 1):
            losses[ep] = []
            with tqdm(total=int(np.floor(n_sim / batch_size)), desc='Training epoch {}'.format(ep)) as p_bar:
                # Loop through dataset
                for bi, batch in enumerate(data_set):
                    # Extract params from batch
                    model_indices_b, sim_data_b = batch[0], batch[1]

                    # One step backprop
                    loss = self._train_step(model_indices_b, sim_data_b)

                    # Store loss and update progress bar
                    losses[ep].append(loss)
                    p_bar.set_postfix_str("Epoch {0},Batch {1},Loss: {2:.3f},Running Loss: {3:.3f}"
                                          .format(ep, bi + 1, loss, np.mean(losses[ep])))
                    p_bar.update(1)

            # Store after each epoch, if specified
            if self.manager is not None:
                self.manager.save()
        return losses

    def train_rounds(self, epochs, rounds, sim_per_round, batch_size, n_obs, **kwargs):
        """
        Trains the inference network(s) via round-based learning.
        ----------
        
        Arguments:
        epochs         : int -- number of epochs (and number of times a checkpoint is stored)
        rounds         : int -- number of rounds to perform 
        sim_per_round  : int -- number of simulations per round
        batch_size     : int -- number of simulations to perform at each backprop step
        ----------

        Returns:
        losses : dict (ep_num : list_of_losses) -- a dictionary storing the losses across epochs and iterations
        """

        # Make sure n_obs is fixed, otherwise not working
        assert type(n_obs) is int, \
            'Round-based training currently only works with fixed n_obs. Use online learning for variable n_obs'

        losses = dict()
        model_indices = sim_data = None
        for r in range(1, rounds + 1):

            # Data generation step
            if r == 1:
                # Initial simulation
                print('Simulating initial {} data sets...'.format(sim_per_round))
                model_indices, sim_data = self._forward_inference(sim_per_round, n_obs, **kwargs)

            else:
                # Further simulations
                print('Simulating new {} data sets and appending to previous...'.format(sim_per_round))
                print('New total number of simulated data sets: {}'.format(sim_per_round * r))
                model_indices_r, sim_data_r = self._forward_inference(sim_per_round, n_obs, **kwargs)

                # Append to previous
                model_indices = np.concatenate((model_indices, model_indices_r), axis=0) \
                    if model_indices is not None else model_indices
                sim_data = np.concatenate((sim_data, sim_data_r), axis=0) \
                    if sim_data is not None else sim_data

            # Train offline with generated data and model indices
            losses_r = self.train_offline(epochs, batch_size, model_indices, sim_data)
            losses[r] = losses_r
        return losses

    def _forward_inference(self, n_sim, n_obs, **_kwargs):
        """
        Performs one step of multi-model forward inference.
        ----------
        
        Arguments:
        n_sim : int -- number of simulation to perform at the given step (i.e., batch size)
        n_obs : int -- number of observations to generate from each forward model
        ----------

        Returns:
        model_indices_oh : np.array (np.float32) of shape (batch_size, n_models)
                            -- array of one-hot-encoded model-indices
        sim_data         : np.array (np.float32) of shape (batch_size, N, data_dim)
                            -- array of simulated data sets
        """

        # Sample model indices, (params), and sim_data
        model_indices_oh, _params, sim_data = self.generative_model(n_sim, n_obs)

        # Compute hand-crafted summary statistics, if given
        if self.summary_stats is not None:
            sim_data = self.summary_stats(sim_data)

        return model_indices_oh, sim_data

    def _train_step(self, model_indices, sim_data):
        """
        Performs one step of backpropagation with the given model indices and data.
        ----------
        
        Arguments:
        model_indices : np.array (np.float32) of shape (batch_size, n_models) -- array of one-hot-encoded model indices
        sim_data      : np.array (np.float32) of shape (batch_size, N, data_dim) -- array of simulated data sets       
        ----------

        Returns:
        loss : tf.Tensor of shape (,), i.e., a scalar representing the average loss over the batch of m and x
        """

        # Compute loss and store gradients
        # Assumes that the loss will call the network with appropriate inputs and outputs
        with tf.GradientTape() as tape:
            loss = self.loss(self.network, model_indices, sim_data)

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

    def load_pretrained_network(self):
        """
        Attempts to load a pre-trained network if checkpoint path is provided and a checkpoint manager exists.
        """

        if self.manager is None or self.checkpoint is None:
            return False
        status = self.checkpoint.restore(self.manager.latest_checkpoint)
        return status


class ParameterEstimationTrainer:

    def __init__(self, network, generative_model: SimpleGenerativeModel, loss=None, summary_stats=None, optimizer=None,
                 learning_rate=0.0005, checkpoint_path=None, max_to_keep=5, clip_method='global_norm', clip_value=None):
        """
        Creates a trainer instance for performing single-model forward inference and training an
        amortized neural estimator for parameter estimation (BayesFlow). If a checkpoint_path is provided, the
        network's weights will be stored after each training epoch. If the folder contains a 
        checkpoint, the trainer will try to load the weights and continue training with a pre-trained net.
        ----------
        
        Arguments:
        network           : bayesflow.Amortizer instance -- the neural architecture to be optimized
        generative_model  : callable -- a function or an object with n_sim and n_obs mandatory arguments 
                            returning randomly sampled parameter vectors and datasets from a process model
        loss              : callable with three mandatory arguments: (network, params, sim_x), if None ('default'), 
                            kl_latent_space used
        ----------
        
        Keyword arguments:
        summary_stats   : callable -- optional summary statistics function
        optimizer       : None or tf.keras.optimizer.Optimizer -- default Adam optimizer (equiv to None) or custom one
        learning_rate   : float -- the learning rate used for the optimizer
        checkpoint_path : string -- optional folder name for storing the trained network
        max_to_keep     : int -- optional number of checkpoints to keep
        clip_method     : string in ('norm', 'value', 'global_norm') -- optional gradient clipping method
        clip_value      : float -- the value used for gradient clipping when clip_method is set to 'value' or 'norm'
        """

        # Basic attributes
        self.network = network
        self.generative_model = generative_model
        # Default or custom loss
        if loss is None:
            self.loss = kl_latent_space
        else:
            self.loss = loss
        # Optional hand-crafted summaries
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
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.network)
            self.manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=max_to_keep)
            self.checkpoint.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Networks loaded from {}".format(self.manager.latest_checkpoint))
            else:
                print("Initializing networks from scratch.")
        else:
            self.checkpoint = None
            self.manager = None
        self.checkpoint_path = checkpoint_path

        # Do some basic preliminary sanity checks
        self._check_consistency()

    def train_experience_replay(self, epochs, batch_size, iterations_per_epoch, capacity, n_obs, **kwargs):
        """
        Trains the inference network(s) via experience replay. 
        
        Additional keyword arguments are passed to the simulators.
        ----------
        
        Arguments:
        epochs               : int -- number of epochs (and number of times a checkpoint is stored)
        batch_size           : int -- number of simulations to perform at each backprop step
        iterations_per_epoch : int -- number of batch simulations to perform per epoch
        capacity               : int -- max number of batches to store in buffer
        n_obs                : int or callable -- if int, then treated as a fixed number of observations,
                               if callable, then treated as a function for sampling N, i.e., N ~ p(N)
        
        ----------

        Returns:
        losses : dict (ep_num : list_of_losses) -- a dictionary storing the losses across epochs and iterations
        """

        # Initialize losses dictionary and memory replay buffer
        losses = dict()
        mem = MemoryReplayBuffer(capacity)

        for ep in range(1, epochs + 1):
            losses[ep] = []
            with tqdm(total=iterations_per_epoch, desc='Training epoch {}'.format(ep)) as p_bar:

                for it in range(1, iterations_per_epoch + 1):

                    # Determine n_obs and generate data on-the-fly
                    if type(n_obs) is int:
                        n_obs_it = n_obs
                    else:
                        n_obs_it = n_obs()
                    # Simulate and add to buffer
                    params, sim_data = self._forward_inference(batch_size, n_obs_it, **kwargs)
                    mem.store(params, sim_data)

                    # Sample from buffer
                    params, sim_data = mem.sample()

                    # One step backprop
                    loss = self._train_step(params, sim_data)

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

    def train_online(self, epochs, iterations_per_epoch, batch_size, n_obs, **kwargs):
        """
        Trains the inference network(s) via online learning. Additional keyword arguments
        are passed to the simulators.
        ----------
        
        Arguments:
        epochs               : int -- number of epochs (and number of times a checkpoint is stored)
        iterations_per_epoch : int -- number of batch simulations to perform per epoch
        batch_size           : int -- number of simulations to perform at each backprop step
        n_obs                : int or callable -- if int, then treated as a fixed number of observations,
                               if callable, then treated as a function for sampling N, i.e., N ~ p(N)
        ----------

        Returns:
        losses : dict (ep_num : list_of_losses) -- a dictionary storing the losses across epochs and iterations
        """

        losses = dict()
        for ep in range(1, epochs + 1):
            losses[ep] = []
            with tqdm(total=iterations_per_epoch, desc='Training epoch {}'.format(ep)) as p_bar:
                for it in range(1, iterations_per_epoch + 1):

                    # Determine n_obs and generate data on-the-fly
                    if type(n_obs) is int:
                        n_obs_it = n_obs
                    else:
                        n_obs_it = n_obs()
                    params, sim_data = self._forward_inference(batch_size, n_obs_it, **kwargs)

                    # One step backprop
                    loss = self._train_step(params, sim_data)

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

    def train_offline(self, epochs, batch_size, params, sim_data):
        """
        Trains the inference network(s) via offline learning. Assume params and data have already 
        been simulated (i.e., forward inference). 
        ----------
        
        Arguments:
        epochs       : int -- number of epochs (and number of times a checkpoint is stored)
        batch_size   : int -- number of simulations to perform at each backprop step
        params       : np.array of shape (n_sim, n_params) -- the true data-generating parameters
        sim_data     : np.array of shape (n_sim, n_obs, data_dim) -- the simulated data sets from each model
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
            .from_tensor_slices((params.astype(np.float32), sim_data.astype(np.float32))) \
            .shuffle(n_sim) \
            .batch(batch_size)

        losses = dict()
        for ep in range(1, epochs + 1):
            losses[ep] = []
            with tqdm(total=int(np.ceil(n_sim / batch_size)), desc='Training epoch {}'.format(ep)) as p_bar:
                # Loop through dataset
                for bi, batch in enumerate(data_set):
                    # Extract params from batch
                    params_b, sim_data_b = batch[0], batch[1]

                    # One step backprop
                    loss = self._train_step(params_b, sim_data_b)

                    # Store loss and update progress bar
                    losses[ep].append(loss)
                    p_bar.set_postfix_str("Epoch {0},Batch {1},Loss: {2:.3f},Running Loss: {3:.3f}"
                                          .format(ep, bi + 1, loss, np.mean(losses[ep])))
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
        assert type(n_obs) is int, \
            'Round-based training currently only works with fixed n_obs. ' \
            'Use online learning for variable n_obs or fix n_obs to an integer value.'

        losses = dict()
        params = sim_data = None
        for r in range(1, rounds + 1):
            # Data generation step
            if r == 1:
                # Simulate initial data
                print('Simulating initial {} data sets...'.format(sim_per_round))
                params, sim_data = self._forward_inference(sim_per_round, n_obs, **kwargs)
            else:
                # Simulate further data
                print('Simulating new {} data sets and appending to previous...'.format(sim_per_round))
                print('New total number of simulated data sets: {}'.format(sim_per_round * r))
                params_r, sim_data_r = self._forward_inference(sim_per_round, n_obs, **kwargs)

                # Add new simulations to previous data.
                params = np.concatenate((params, params_r), axis=0) if params is not None else params_r
                sim_data = np.concatenate((sim_data, sim_data_r), axis=0) if sim_data is not None else sim_data

            # Train offline with generated stuff
            losses_r = self.train_offline(epochs, batch_size, params, sim_data)
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
        assert type(n_obs) is int, \
            'Offline training currently only works with fixed n_obs. ' \
            'Use online learning for variable n_obs or fix n_obs to an integer value.'

        # Simulate data
        print('Simulating {} data sets upfront...'.format(n_sim))
        params, sim_data = self._forward_inference(n_sim, n_obs, summarize=False, **kwargs)

        # Train offline
        losses = self.train_offline(epochs, batch_size, params, sim_data)
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

        Keyword arguments:
        summarize : bool -- whether to summarize the data if hand-crafted summaries are given

        Returns:
        params    : np.array (np.float32) of shape (batch_size, param_dim) -- array of sampled parameters
        sim_data  : np.array (np.float32) of shape (batch_size, n_obs, data_dim) -- array of simulated data sets
        """

        # Simulate data with n_sims and n_obs
        # Return shape of params is (batch_size, param_dim)
        # Return shape of data is (batch_size, n_obs, data_dim)
        params, sim_data = self.generative_model(n_sim, n_obs, **kwargs)

        # Compute hand-crafted summary stats, if given
        if summarize and self.summary_stats is not None:
            # Return shape in this case is (batch_size, n_sum)
            sim_data = self.summary_stats(sim_data)

        return params.astype(np.float32), sim_data.astype(np.float32)

    def _train_step(self, params, sim_data):
        """
        Performs one step of backpropagation with the given model indices and data.
        ----------
        
        Arguments:
        params    : np.array (np.float32) of shape (batch_size, n_params) -- matrix of n_samples x n_params
        sim_data  : np.array (np.float32) of shape (batch_size, n_obs, data_dim) or (batch_size, summary_dim) 
                    -- array of simulated data sets (or summary statistics thereof)      
        ----------

        Returns:
        loss : tf.Tensor of shape (,), i.e., a scalar representing the average loss over the batch of m and x
        """

        # Compute loss and store gradients
        with tf.GradientTape() as tape:
            loss = self.loss(self.network, params, sim_data)

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

    def _check_consistency(self):
        """
        Tests whether everything works as expected.
        """

        # Run forward inference with n_sim=2 and catch any exception
        try:
            _, sim_data = self._forward_inference(n_sim=2, n_obs=150)
        except Exception as err:
            raise SimulationError(repr(err))

        # Run summary network check
        if self.summary_stats is not None:
            try:
                _ = self.summary_stats(sim_data)
            except Exception as err:
                raise SummaryStatsError(repr(err))

        # TODO: Run checks whether the network works with the data format

        # TODO: Run checks that loss works with the provided network
