from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tqdm.notebook import tqdm

from bayesflow.buffer import MemoryReplayBuffer
from bayesflow.exceptions import SimulationError, SummaryStatsError, OperationNotSupportedError, LossError
from bayesflow.helpers import clip_gradients
from bayesflow.losses import kl_latent_space, log_loss


class BaseTrainer(ABC):

    def __init__(self, network, generative_model, loss, summary_stats, optimizer,
                 learning_rate, checkpoint_path, max_to_keep, clip_method, clip_value):
        """Base class for a trainer performing forward inference and training an amortized neural estimator.

        Parameters
        ----------
        network         : bayesflow.amortizers.Amortizer
            The neural architecture to be optimized
        generative_model : bayesflow.models.GenerativeModel
            A generative model returning randomly sampled parameter vectors and datasets from a process model
        loss            : callable
            Loss function with three arguments: (network, m_indices, x)
        summary_stats   : callable
            Optional summary statistics function
        optimizer       : None or tf.keras.optimizer.Optimizer
            Optimizer for the neural network. ``None`` will result in `tf.keras.optimizers.Adam`
        learning_rate   : float
            The learning rate used for the optimizer
        checkpoint_path : string, optional
            Optional folder name for storing the trained network
        max_to_keep     : int, optional
            Number of checkpoints to keep
        clip_method     : {'norm', 'value', 'global_norm'}
            Optional gradient clipping method
        clip_value      : float
            The value used for gradient clipping when clip_method is in {'value', 'norm'}
        """

        self.network = network

        self.generative_model = generative_model
        if self.generative_model is None:
            print("TRAINER INITIALIZATION: No generative model provided. Only offline learning mode is available!")

        # subclass handles default loss
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

        self._check_consistency()

    def load_pretrained_network(self):
        """Attempts to load a pre-trained network if checkpoint path is provided and a checkpoint manager exists.
        """

        if self.manager is None or self.checkpoint is None:
            return False
        status = self.checkpoint.restore(self.manager.latest_checkpoint)
        return status

    def train_online(self, epochs, iterations_per_epoch, batch_size, n_obs, **kwargs):
        """Trains the inference network(s) via online learning. Additional keyword arguments
        are passed to the simulators.

        Parameters
        ----------
        epochs               : int -- number of epochs (and number of times a checkpoint is stored)
        iterations_per_epoch : int -- number of batch simulations to perform per epoch
        batch_size           : int -- number of simulations to perform at each backprop step
        n_obs : int or callable
            Number of observations for each simulated dataset.

            -  if `int`, then treated as a fixed number of observations, \n
            -  if `callable`, then treated as a function for sampling N, i.e., :math:`N \sim p(N)`

        **kwargs : dict
            Passed to the simulator(s)

        Returns
        -------
        losses : dict(ep_num : list(losses))
            A dictionary storing the losses across epochs and iterations
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
                    args = self._forward_inference(batch_size, n_obs_it, **kwargs)

                    # One step backprop
                    loss = self._train_step(*args)

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

    def train_offline(self, epochs, batch_size, *args, **kwargs):
        """Trains the inference network(s) via offline learning. Assume params and data have already
        been simulated (i.e., forward inference).

        Parameters
        ----------
        epochs           : int
            Number of epochs (and number of times a checkpoint is stored)
        batch_size       : int
            Number of simulations to perform at each backpropagation step
        *args : tuple
            Input to the trainer, e.g. (params, sim_data) or (model_indices, params, sim_data)
        **kwargs: dict(arg_name, arg)
            Input to the trainer, e.g. {'params': theta, 'sim_data': x}
            Note that argument names must be in {'model_indices', 'params', 'sim_data'}

        Returns
        -------
        losses : dict(ep_num : list(losses))
            A dictionary storing the losses across epochs and iterations

        Important
        ---------

        -  If you use `args`, the last entry of ``args`` must be your simulated data!
        -  If you use `kwargs`, the order of the ``kwargs`` inputs does not matter.
           Please use the keyword names in {'model_indices', 'params', 'sim_data'}

        Examples
        --------
        Parameter estimation (args)

        >>> true_params, sim_data = simple_generative_model(n_sim=1000, n_obs=100)
        >>> trainer.train_offline(10, 32, true_params, sim_data)

        Model comparison (args)

        >>> true_model_indices, _, sim_data = meta_generative_model(n_sim=1000, n_obs=100)
        >>> trainer.train_offline(10, 32, true_model_indices, sim_data)

        Meta (args)

        >>> true_model_indices, true_params, sim_data = meta_generative_model(n_sim=1000, n_obs=100)
        >>> trainer.train_offline(10, 32, true_model_indices, true_params, sim_data)

        Parameter estimation (keyword-args)

        >>> true_params, sim_data = simple_generative_model(n_sim=1000, n_obs=100)
        >>> trainer.train_offline(epochs=10, batch_size=32, params=true_params, sim_data=sim_data)

        Model comparison (keyword-args)

        >>> true_model_indices, _, sim_data = meta_generative_model(n_sim=1000, n_obs=100)
        >>> trainer.train_offline(epochs=10, batch_size=32, model_indices=true_model_indices, sim_data=sim_data)

        Meta (keyword-args)

        >>> true_model_indices, true_params, sim_data = meta_generative_model(n_sim=1000, n_obs=100)
        >>> trainer.train_offline(epochs=10, batch_size=32,
        ...                       params=true_params, model_indices=true_model_indices, sim_data=sim_data)
        """

        # preprocess kwargs to args
        args = self._train_offline_kwargs_to_args(args, kwargs)

        # Convert to a data set
        n_sim = int(args[-1].shape[0])

        # Compute summary statistics, if provided
        if self.summary_stats is not None:
            print('Computing hand-crafted summary statistics...')
            args = list(args)
            args[-1] = self.summary_stats(args[-1])
            args = tuple(args)

        print('Converting {} simulations to a TensorFlow data set...'.format(n_sim))
        data_set = tf.data.Dataset \
            .from_tensor_slices(args) \
            .shuffle(n_sim) \
            .batch(batch_size)

        losses = dict()
        for ep in range(1, epochs + 1):
            losses[ep] = []
            with tqdm(total=int(np.ceil(n_sim / batch_size)), desc='Training epoch {}'.format(ep)) as p_bar:
                # Loop through dataset
                for bi, batch in enumerate(data_set):
                    # Extract arguments from batch
                    args_b = tuple(batch)

                    # One step backpropagation
                    loss = self._train_step(*args_b)

                    # Store loss and update progress bar
                    losses[ep].append(loss)
                    p_bar.set_postfix_str("Epoch {0},Batch {1},Loss: {2:.3f},Running Loss: {3:.3f}"
                                          .format(ep, bi + 1, loss, np.mean(losses[ep])))
                    p_bar.update(1)

            # Store after each epoch, if specified
            if self.manager is not None:
                self.manager.save()
        return losses

    def simulate_and_train_offline(self, n_sim, epochs, batch_size, n_obs, **kwargs):
        """Simulates n_sim data sets from _forward_inference and then trains the inference network(s)
        via offline learning.

        Parameters
        ----------
        n_sim          : int
            Total number of simulations to perform
        epochs         : int
            Number of epochs (and number of times a checkpoint is stored)
        batch_size     : int
            Number of simulations to perform at each backprop step
        n_obs          : int
            Number of observations for each dataset
        **kwargs : dict
            Passed to the simulator(s)

        Returns
        -------
        losses : dict(ep_num : list(losses))
            A dictionary storing the losses across epochs and iterations
        """

        # Make sure n_obs is fixed, otherwise not working, for now
        assert type(n_obs) is int, \
            'Offline training currently only works with fixed n_obs. ' \
            'Use online learning for variable n_obs or fix n_obs to an integer value.'

        # Simulate data
        print('Simulating {} data sets upfront...'.format(n_sim))
        args = self._forward_inference(n_sim, n_obs, summarize=False, **kwargs)

        # Train offline
        losses = self.train_offline(epochs, batch_size, *args)
        return losses

    def train_rounds(self, epochs, rounds, sim_per_round, batch_size, n_obs, **kwargs):
        """Trains the inference network(s) via round-based learning.

        Parameters
        ----------
        epochs         : int
            Number of epochs (and number of times a checkpoint is stored)
        rounds         : int
            Number of rounds to perform
        sim_per_round  : int
            Number of simulations per round
        batch_size     : int
            Number of simulations to perform at each backpropagation step
        n_obs          : int
            Number of observations (fixed) for each data set
        **kwargs : dict
            Passed to the simulator(s)

        Returns
        -------
        losses : dict(ep_num : list(losses))
            A dictionary storing the losses across epochs and iterations
        """

        # Make sure n_obs is fixed, otherwise not working
        assert type(n_obs) is int, \
            'Round-based training currently only works with fixed n_obs. ' \
            'Use online learning for variable n_obs or fix n_obs to an integer value.'
        losses = dict()
        args = None
        first_round = True
        for r in range(1, rounds + 1):
            # Data generation step
            if first_round:
                # Simulate initial data
                print('Simulating initial {} data sets...'.format(sim_per_round))
                args = self._forward_inference(sim_per_round, n_obs, **kwargs)
                first_round = False
            else:
                # Simulate further data
                print('Simulating new {} data sets and appending to previous...'.format(sim_per_round))
                print('New total number of simulated data sets: {}'.format(sim_per_round * r))
                args_r = self._forward_inference(sim_per_round, n_obs, **kwargs)

                # Add new simulations to previous data.
                args = list(args) if args is not None else []
                for i in range(len(args)):
                    args[i] = np.concatenate((args[i], args_r[i]), axis=0)
                args = tuple(args)

            # Train offline with generated stuff
            losses_r = self.train_offline(epochs, batch_size, *args)
            losses[r] = losses_r

        return losses

    def _train_step(self, *args):
        """Computes loss and applies gradients.
        """
        with tf.GradientTape() as tape:
            loss = self.loss(self.network, *args)

        # One step backprop
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self._apply_gradients(gradients, self.network.trainable_variables)

        return loss.numpy()

    def _apply_gradients(self, gradients, tensors):
        """Updates each tensor in the 'variables' list via backpropagation. Operation is performed in-place.

        Parameters
        ----------
        gradients: list(tf.Tensor)
            The list of gradients for all neural network parameters
        tensors: list(tf.Tensor)
            The list of all neural network parameters
        """

        # Optional gradient clipping
        if self.clip_value is not None:
            gradients = clip_gradients(gradients, clip_value=self.clip_value, clip_method=self.clip_method)
        self.optimizer.apply_gradients(zip(gradients, tensors))

    @abstractmethod
    def _forward_inference(self, n_sim, n_obs, **kwargs):
        """Simulate arguments for training (abstract method).

        In subclasses, this method is implemented as:

        -  (params, sim_data) for :class:'ParameterEstimationTrainer'
        -  (model_indices, sim_data) for :class:'ModelComparisonTrainer'
        -  (model_indices, params, sim_data) for :class:'MetaTrainer'
        """
        raise NotImplementedError

    def _train_offline_kwargs_to_args(self, args, kwargs):
        """Unifies signature of trainer.train_offline to work with *args or **kwargs

        Parameters
        ----------
        args: tuple
            List of non-keyword arguments
        kwargs: dict
            List of keyword-arguments

        Returns
        -------
        args: tuple
            Preprocessed tuple for train_offline

        """

        if not args and not kwargs:
            raise OperationNotSupportedError("Must provide inputs (e.g. params, sim_data)!")

        if args and kwargs:
            raise OperationNotSupportedError("Please give all arguments with keyword or all arguments without keyword!")

        if not args and kwargs:
            args = []
            if 'model_indices' in kwargs.keys():
                args.append(kwargs.pop('model_indices'))
            if 'params' in kwargs.keys():
                args.append(kwargs.pop('params'))
            args.append(kwargs.pop('sim_data'))

            args = tuple(args)

        return args

    def _check_consistency(self):
        """Tests whether everything works as expected after initialization
        """
        if self.generative_model is None:
            return

        # Run forward inference with n_sim=2 and catch any exception
        try:
            args = self._forward_inference(n_sim=2, n_obs=150)
        except Exception as err:
            raise SimulationError(repr(err))

        # Run summary network check
        if self.summary_stats is not None:
            try:
                _ = self.summary_stats(args[-1])
            except Exception as err:
                raise SummaryStatsError(repr(err))

        # Run loss function check
        try:
            args = self._forward_inference(n_sim=2, n_obs=150)
            _loss = self.loss(self.network, *args)
        except Exception as err:
            raise LossError(repr(err))


class MetaTrainer(BaseTrainer):

    def __init__(self, network, generative_model=None, loss=None, summary_stats=None, optimizer=None,
                 learning_rate=0.0005, checkpoint_path=None, max_to_keep=5, clip_method='global_norm', clip_value=None):
        """ Creates a trainer instance for performing multi-model forward inference and training an
        amortized neural estimator for parameter estimation and model comparison (BayesFlow).

        If a checkpoint_path is provided, the network's weights will be stored after each training epoch.
        If the folder contains a checkpoint, the trainer will try to load the weights and continue
        training with a pre-trained net.
        """

        # Default or custom loss
        if loss is None:
            _loss = kl_latent_space
        else:
            _loss = loss

        super().__init__(network, generative_model, _loss, summary_stats, optimizer, learning_rate,
                         checkpoint_path, max_to_keep, clip_method, clip_value)

    def _forward_inference(self, n_sim, n_obs, summarize=True, **kwargs):
        """Performs one step of multi-model forward inference.

        Parameters
        ----------
        n_sim : int
            Number of simulations to perform at the given step (i.e., batch size)
        n_obs : int or callable
            Number of observations for each simulated dataset.

            -  if `int`, then treated as a fixed number of observations, \n
            -  if `callable`, then treated as a function for sampling N, i.e., :math:`N \sim p(N)`

        summarize : bool, default:True
            Whether to summarize the data if hand-crafted summaries are given

        Returns
        -------
        model_indices: np.array(np.float32)
            One-hot encoded model indices, shape (batch_size, n_models)
        params    : np.array(np.float32)
            array of sampled parameters, shape (batch_size, param_dim)
        sim_data  : np.array(np.float32)
            array of simulated data sets, shape (batch_size, n_obs, data_dim)

        Raises
        ------
        OperationNotSupportedError
            If the trainer has no generative model but `trainer._forward_inference`
            is called (i.e. needs to simulate data from the generative model)
        """

        if self.generative_model is None:
            raise OperationNotSupportedError("No generative model specified. Only offline learning is available!")

        # Simulate data with n_sims and n_obs
        # Return shape of model_indices is (batch_size, n_models)
        # Return shape of params is (batch_size, param_dim)
        # Return shape of data is (batch_size, n_obs, data_dim)
        model_indices, params, sim_data = self.generative_model(n_sim, n_obs, **kwargs)

        # Compute hand-crafted summary stats, if given
        if summarize and self.summary_stats is not None:
            # Return shape in this case is (batch_size, n_sum)
            sim_data = self.summary_stats(sim_data)

        return model_indices, params, sim_data


class ModelComparisonTrainer(BaseTrainer):

    def __init__(self, network, generative_model=None, loss=None, summary_stats=None, optimizer=None, n_models=None,
                 learning_rate=0.0005, checkpoint_path=None, max_to_keep=5, clip_method='global_norm', clip_value=None):
        """Creates a trainer instance for performing multi-model forward inference and training an
        amortized neural estimator for model comparison.

        If a checkpoint_path is provided, the network's weights will be stored after each training epoch.
        If the folder contains a checkpoint, the trainer will try to load the weights and continue training with
        a pre-trained net.
        """

        # Default or custom loss
        if loss is None:
            _loss = log_loss
        else:
            _loss = loss

        self.n_models = n_models
        super().__init__(network, generative_model, _loss, summary_stats, optimizer, learning_rate,
                         checkpoint_path, max_to_keep, clip_method, clip_value)

    def train_offline(self, epochs, batch_size, *args, **kwargs):
        """Handles one-hot encoding if necessary and calls superclass method.

        Trains the inference network(s) via offline learning. Assume params and data have already
        been simulated (i.e., forward inference).

        Parameters
        ----------
        epochs           : int
            Number of epochs (and number of times a checkpoint is stored)
        batch_size       : int
            Number of simulations to perform at each backpropagation step
        *args : tuple
            Input to the trainer: (model_indices, sim_data)
        **kwargs: dict(arg_name, arg)
            Input to the trainer, {'model_indices': m_oh, 'sim_data': x}
            Note that argument names must be in {'model_indices', 'sim_data'}

        Returns
        -------
        losses : dict(ep_num : list(losses))
            A dictionary storing the losses across epochs and iterations

        Important
        ---------

        -  If you use `args`, the last entry of ``args`` must be your simulated data!
        -  If you use `kwargs`, the order of the ``kwargs`` inputs does not matter.
           Please use the keyword names in {'model_indices', 'sim_data'}
        """

        args = self._train_offline_kwargs_to_args(args, kwargs)

        # Handle automated one-hot encoding
        if len(args) == 2:
            (model_indices, sim_data), n_models = args, None
        elif len(args) == 3:
            model_indices, sim_data, n_models = args
        else:
            raise OperationNotSupportedError()

        if len(model_indices.shape) == 1:
            if n_models is None:
                if self.n_models is None:
                    print("No n_models provided but model indices are 1D. "
                          "Assuming len(np.unique(model_indices))=n_models.")
                    n_models = len(np.unique(model_indices))
                    print("Saving n_models in the trainer.")
                    self.n_models = n_models
                else:
                    n_models = self.n_models

            print('One-hot-encoding model indices...')
            model_indices = to_categorical(model_indices, num_classes=n_models)

        # call train_offline of superclass with one-hot encoded model_indices
        super().train_offline(epochs, batch_size, model_indices, sim_data)

    def _forward_inference(self, n_sim, n_obs, **kwargs):
        """Performs one step of multi-model forward inference.

        Parameters
        ----------
        n_sim : int
            Number of simulations to perform at the given step (i.e., batch size)
        n_obs : int or callable
            Number of observations for each simulated dataset.

            -  if `int`, then treated as a fixed number of observations, \n
            -  if `callable`, then treated as a function for sampling N, i.e., :math:`N \sim p(N)`

        summarize : bool, default:True
            Whether to summarize the data if hand-crafted summaries are given

        Returns
        -------
        model_indices: np.array(np.float32)
            One-hot encoded model indices, shape (batch_size, n_models)
        sim_data  : np.array(np.float32)
            array of simulated data sets, shape (batch_size, n_obs, data_dim)

        Raises
        ------
        OperationNotSupportedError
            If the trainer has no generative model but `trainer._forward_inference`
            is called (i.e. needs to simulate data from the generative model)
        """

        if self.generative_model is None:
            raise OperationNotSupportedError("No generative model specified. Only offline learning is available!")

        # Sample model indices, (params), and sim_data
        model_indices_oh, _params, sim_data = self.generative_model(n_sim, n_obs, **kwargs)

        # Compute hand-crafted summary statistics, if given
        if self.summary_stats is not None:
            sim_data = self.summary_stats(sim_data)

        return model_indices_oh, sim_data


class ParameterEstimationTrainer(BaseTrainer):

    def __init__(self, network, generative_model=None, loss=None, summary_stats=None, optimizer=None,
                 learning_rate=0.0005, checkpoint_path=None, max_to_keep=5, clip_method='global_norm', clip_value=None):
        """Creates a trainer instance for performing single-model forward inference and training an
        amortized neural estimator for parameter estimation (BayesFlow).

        If a checkpoint_path is provided, the network's weights will be stored after each training epoch.
        If the folder contains a  checkpoint, the trainer will try to load the weights and continue training
        with a pre-trained net.
        """

        if loss is None:
            _loss = kl_latent_space
        else:
            _loss = loss

        super().__init__(network, generative_model, _loss, summary_stats, optimizer, learning_rate,
                         checkpoint_path, max_to_keep, clip_method, clip_value)

    def train_experience_replay(self, epochs, batch_size, iterations_per_epoch, capacity, n_obs, **kwargs):
        """Trains the inference network(s) via experience replay.
        
        Parameters
        ----------
        epochs               : int
            Number of epochs (and number of times a checkpoint is stored)
        batch_size           : int
            Number of simulations to perform at each backpropagation step
        iterations_per_epoch : int
            Number of batch simulations to perform per epoch
        capacity               : int
            Max number of batches to store in buffer
        n_obs : int or callable
            Number of observations for each simulated dataset.

            -  if `int`, then treated as a fixed number of observations, \n
            -  if `callable`, then treated as a function for sampling N, i.e., :math:`N \sim p(N)`


        Returns
        -------
        losses : dict(ep_num : list(losses))
            A dictionary storing the losses across epochs and iterations
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

    def _forward_inference(self, n_sim, n_obs, summarize=True, **kwargs):
        """
        Performs one step of single-model forward inference.

        Parameters
        ----------
        n_sim : int
            Number of simulations to perform at the given step (i.e., batch size)
        n_obs : int or callable
            Number of observations for each simulated dataset.

            -  if `int`, then treated as a fixed number of observations, \n
            -  if `callable`, then treated as a function for sampling N, i.e., :math:`N \sim p(N)`

        summarize : bool, default:True
            Whether to summarize the data if hand-crafted summaries are given

        Returns
        -------
        params    : np.array(np.float32)
            array of sampled parameters, shape (batch_size, param_dim)
        sim_data  : np.array(np.float32)
            array of simulated data sets, shape (batch_size, n_obs, data_dim)

        Raises
        ------
        OperationNotSupportedError
            If the trainer has no generative model but `trainer._forward_inference`
            is called (i.e. needs to simulate data from the generative model)

        """

        if self.generative_model is None:
            raise OperationNotSupportedError("No generative model specified. Only offline learning is available!")

        # Simulate data with n_sims and n_obs
        # Return shape of params is (batch_size, param_dim)
        # Return shape of data is (batch_size, n_obs, data_dim)
        params, sim_data = self.generative_model(n_sim, n_obs, **kwargs)

        # TODO - Apply transforms, if given

        # Compute hand-crafted summary stats, if given
        if summarize and self.summary_stats is not None:
            # Return shape in this case is (batch_size, n_sum)
            sim_data = self.summary_stats(sim_data)

        return params.astype(np.float32), sim_data.astype(np.float32)
