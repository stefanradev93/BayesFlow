# Copyright (c) 2022 The BayesFlow Developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
import numpy as np
import os
from pickle import load as pickle_load
from tqdm.autonotebook import tqdm

import logging

from bayesflow.forward_inference import GenerativeModel, MultiGenerativeModel
logging.basicConfig()

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from bayesflow.configuration import *
from bayesflow.exceptions import SimulationError
from bayesflow.helper_functions import format_loss_string
from bayesflow.helper_classes import SimulationDataset, LossHistory, SimulationMemory, RegressionLRAdjuster
from bayesflow.default_settings import STRING_CONFIGS, DEFAULT_KEYS, OPTIMIZER_DEFAULTS
from bayesflow.amortized_inference import AmortizedLikelihood, AmortizedPosterior, JointAmortizer, ModelComparisonAmortizer
from bayesflow.diagnostics import plot_sbc_histograms, plot_latent_space_2d


class Trainer:
    """ This class connects a generative model (or, already simulated data from a model) with
    a configurator and a neural inference architecture for amortized inference (amortizer). A Trainer 
    instance is responsible for optimizing the amortizer via various forms of simulation-based training.

    At the very minimum, the trainer must be initialized with an `amortizer` instance, which is capable
    of processing the (configured) outputs of a generative model. A `configurator` will then process
    the outputs of the generative model and convert them into suitable inputs for the amortizer. Users
    can choose from a palette of default configurators or create their own configurators, essentially
    building a modularized pipeline GenerativeModel -> Configurator -> Amortizer. Most complex models 
    wtill require custom configurators.

    Currently, the trainer supports the following simulation-based training regimes, based on efficiency
    considerations:

    - Online training
        Usage:
        >>> trainer.train_online(self, epochs, iterations_per_epoch, batch_size, **kwargs)

        This training regime is optimal for fast generative models which can efficiently simulated data on-the-fly.
        In order for this training regime to be efficient, on-the-fly batch simulations should not take longer than 2-3 seconds.
        
        Important: overfitting presents a danger when using a small simulated data set, so it is recommended to use
        some amount of regularization for the neural amortizer.
    
    - Round-based training
        Usage:
        >>> trainer.train_rounds(self, rounds, sim_per_round, epochs, batch_size, **kwargs)

        This training regime is optimal for slow, but still reasonably performant generative models.
        In order for this training regime to be efficient, on-the-fly batch simulations should not take longer than one 2-3 minutes.

    - Offline taining
        Usage:
        >>> trainer.train_offline(self, simulations_dict, epochs, batch_size, **kwargs)

        This training regime is optimal for very slow, external simulators, which take several minutes for a single simulation.
        It assumes that all training data has been already simulated and stored on disk.
    
    Note: For extremely slow simulators (i.e., more than an hour of a single simulation), the BayesFlow framework might not be the ideal
    choice and should probably be considered in combination with a black-box surrogate optimization method, such as Bayesian optimization.
    """

    def __init__(self, amortizer, generative_model=None, configurator=None, optimizer=None,
                 learning_rate=0.0005, checkpoint_path=None, max_to_keep=3, skip_checks=False, 
                 memory=True, optional_stopping=False, **kwargs):
        """Creates a trainer which will use a generative model (or data simulated from it) to optimize
        a neural arhcitecture (amortizer) for amortized posterior inference, likelihood inference, or both.

        Parameters
        ----------
        amortizer         : bayesflow.amortizers.Amortizer
            The neural architecture to be optimized
        generative_model  : bayesflow.forward_inference.GenerativeModel
            A generative model returning a dictionary with randomly sampled parameters, data, and optional context
        configurator      : callable 
            A callable object transforming and combining the outputs of the generative model into inputs for BayesFlow
        optimizer         : tf.keras.optimizer.Optimizer or None
            Optimizer for the neural network. `None` will result in `tf.keras.optimizers.Adam`
        learning_rate     : float or tf.keras.schedules.LearningRateSchedule
            The learning rate used for the optimizer. Should not be part of the `optimizer_kwargs!` 
        checkpoint_path   : string, optional
            Optional folder name for storing the trained network
        max_to_keep       : int, optional
            Number of checkpoints to keep
        skip_checks       : boolean
            If True, do not perform consistency checks, i.e., simulator runs and passed through nets
        memory            : boolean or bayesflow.SimulationMemory
            If True, store a pre-defined amount of simulations for later use (validation, etc.). 
            If SimulationMemory instance provided, stores a reference to the instance. 
            Otherwise the corresponding attribute will be set to None.
        optional_stopping : boolean, optional, default: False
            Whether to use optional stopping or not during training. Could speed up training.
        **kwargs          : dict, optional, default: {}
            Optional keyword arguments for controling the behavior of the Trainer instance. As of now, these could be:

            optimizer_kwargs         : dict
                Keyword arguments to be passed to the optimizer instance.
            memory_kwargs            : dict
                Keyword arguments to be passed to the `SimulationMemory` instance, if memory=True
            optional_stopping_kwargs : dict
                Keyword arguments to be passed to the `RegressionLRAdjuster` instance if optional_stopping=True
        """

        # Set-up logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        self.amortizer = amortizer
        self.generative_model = generative_model
        if self.generative_model is None:
            logger.info("Trainer initialization: No generative model provided. Only offline learning mode is available!")

        # Determine n models in case model comparison mode
        if type(generative_model) is MultiGenerativeModel:
            _n_models = generative_model.n_models
        elif type(amortizer) is ModelComparisonAmortizer:
            _n_models = amortizer.n_models
        else:
            _n_models = kwargs.get('n_models') 

        # Set-up configurator
        self.configurator = self._manage_configurator(configurator, n_models=_n_models)
        
        # Optimizer settings
        opt_kwargs = kwargs.pop('optimizer_kwargs', {})
        if not opt_kwargs:
            opt_kwargs = OPTIMIZER_DEFAULTS
        if optimizer is None:
            self.optimizer = Adam(learning_rate=learning_rate, **opt_kwargs)
        else:
            self.optimizer = optimizer(learning_rate=learning_rate, **opt_kwargs)

        # Set-up memory classes
        self.loss_history = LossHistory()
        if memory is True:
            self.simulation_memory = SimulationMemory(**kwargs.pop('memory_kwargs', {}))
        elif type(memory) is SimulationMemory:
            self.simulation_memory = memory
        else:
            self.simulation_memory = None

        # Set-up regression learning rate adjuster
        if optional_stopping:
            self.lr_adjuster = RegressionLRAdjuster(self.optimizer, **kwargs.pop('optional_stopping_kwargs', {}))
        else:
            self.lr_adjuster = None

        # Checkpoint and helper classes settings
        self.max_to_keep = max_to_keep
        if checkpoint_path is not None:
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.amortizer)
            self.manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=max_to_keep)
            self.checkpoint.restore(self.manager.latest_checkpoint)
            self.loss_history.load_from_file(checkpoint_path)
            if self.simulation_memory is not None:
                self.simulation_memory.load_from_file(checkpoint_path)
            if self.lr_adjuster is not None:
                self.lr_adjuster.load_from_file(checkpoint_path)
            if self.manager.latest_checkpoint:
                logger.info("Networks loaded from {}".format(self.manager.latest_checkpoint))
            else:
                logger.info("Initialized networks from scratch.")
        else:
            self.checkpoint = None
            self.manager = None
        self.checkpoint_path = checkpoint_path

        # Perform a sanity check wiuth provided components
        if not skip_checks:
            self._check_consistency()

    def diagnose_latent2d(self, inputs=None, **kwargs):
        """ Performs visual pre-inference diagnostics of latent space on either provided validation data
        (new simulations) or internal simulation memory.
        If `inputs is not None`, then diagnostics will be performed on the inputs, regardless
        whether the `simulation_memory` of the trainer is empty or not. If `inputs is None`, then
        the trainer will try to access is memory or raise a `ConfigurationError`.
        
        Parameters
        ----------
        inputs : None, list or dict, optional (default - None)
            The optional inputs to use 
        **kwargs             : dict, optional
            Optional keyword arguments, which could be one of:
            `conf_args`  - optional keyword arguments passed to the configurator
            `net_args`   - optional keyword arguments passed to the amortizer
            `plot_args`  - optional keyword arguments passed to `plot_latent_space_2d`

        Returns
        -------
        losses : dict(ep_num : list(losses))
            A dictionary storing the losses across epochs and iterations
        """

        if type(self.amortizer) is AmortizedPosterior:
            # If no inputs, try memory and throw if no memory
            if inputs is None:
                if self.simulation_memory is None:
                    raise ConfigurationError("You should either enable `simulation memory` or supply the `inputs` argument.")
                else:
                    inputs = self.simulation_memory.get_memory()
            else:
                inputs = self.configurator(inputs, **kwargs.pop('conf_args', {}))
            
            # Do inference
            if type(inputs) is list:
                z, _ = self.amortizer.call_loop(inputs, **kwargs.pop('net_args', {}))
            else:
                z, _ = self.amortizer(inputs, **kwargs.pop('net_args', {}))
            return plot_latent_space_2d(z, **kwargs.pop('plot_args', {}))
        else:
            raise NotImplementedError("Latent space diagnostics are only available for type AmortizedPosterior!")

    def diagnose_sbc_histograms(self, inputs=None, n_samples=None, **kwargs):
        """ Performs visual pre-inference diagnostics via simulation-based calibration (SBC)
        (new simulations) or internal simulation memory.
        If `inputs is not None`, then diagnostics will be performed on the inputs, regardless
        whether the `simulation_memory` of the trainer is empty or not. If `inputs is None`, then
        the trainer will try to access is memory or raise a `ConfigurationError`.
        
        Parameters
        ----------
        inputs    : None, list or dict, optional (default - None)
            The optional inputs to use 
        n_samples : int, optional (default - None)
            The number of posterior samples to draw for each simulated data set.
            If None, the number will be heuristically determined so n_sim / n_draws ~= 20
        **kwargs  : dict, optional
            Optional keyword arguments, which could be one of:
            `conf_args`  - optional keyword arguments passed to the configurator
            `net_args`   - optional keyword arguments passed to the amortizer
            `plot_args`  - optional keyword arguments passed to `plot_sbc`

        Returns
        -------
        losses : dict(ep_num : list(losses))
            A dictionary storing the losses across epochs and iterations
        """

        if type(self.amortizer) is AmortizedPosterior:
            # If no inputs, try memory and throw if no memory
            if inputs is None:
                if self.simulation_memory is None:
                    raise ConfigurationError("You should either ")
                else:
                    inputs = self.simulation_memory.get_memory()
            else:
                inputs = self.configurator(inputs, **kwargs.pop('conf_args', {}))

            # Heuristically determine the number of posterior samples
            if n_samples is None:
                if type(inputs) is list:
                    n_sim = np.sum([inp['parameters'].shape[0] for inp in inputs])
                    n_samples = int(np.ceil(n_sim / 20))
                else:
                    n_samples = int(np.ceil(inputs['parameters'].shape[0] / 20))
                
            # Do inference
            if type(inputs) is list:
                post_samples = self.amortizer.sample_loop(inputs, n_samples=n_samples, **kwargs.pop('net_args', {}))
                prior_samples = np.concatenate([inp['parameters'] for inp in inputs], axis=0)
            else:
                post_samples = self.amortizer(inputs, n_samples, n_samples, **kwargs.pop('net_args', {}))
                prior_samples = inputs['parameters']

            # Check for prior names and override keyword if available
            plot_kwargs = kwargs.pop('plot_args', {})
            if type(self.generative_model) is GenerativeModel:
                plot_kwargs['param_names'] = self.generative_model.param_names
            
            return plot_sbc_histograms(post_samples, prior_samples, **plot_kwargs)
        else:
            raise NotImplementedError("SBC diagnostics are only available for type AmortizedPosterior!")
        
    def load_pretrained_network(self):
        """ Attempts to load a pre-trained network if checkpoint path is provided and a checkpoint manager exists.
        """

        if self.manager is None or self.checkpoint is None:
            return False
        status = self.checkpoint.restore(self.manager.latest_checkpoint)
        return status

    def train_online(self, epochs, iterations_per_epoch, batch_size, save_checkpoint=True, **kwargs):
        """ Trains an amortizer via online learning. Additional keyword arguments
        are passed to the generative mode, configurator, and amortizer.

        Parameters
        ----------
        epochs               : int 
            Number of epochs (and number of times a checkpoint is stored)
        iterations_per_epoch : int 
            Number of batch simulations to perform per epoch
        batch_size           : int 
            Number of simulations to perform at each backprop step
        save_checkpoint      : bool (default - True)
            A flag to decide whether to save checkpoints after each epoch,
            if a checkpoint_path provided during initialization, otherwise ignored
        **kwargs             : dict, optional
            Optional keyword arguments, which can be one of:
            `model_args` - optional keyword arguments passed to the generative model
            `conf_args`  - optional keyword arguments passed to the configurator
            `net_args`   - optional keyword arguments passed to the amortizer

        Returns
        -------
        losses : dict(ep_num : list(losses))
            A dictionary storing the losses across epochs and iterations
        """

        self.loss_history.start_new_run()
        for ep in range(1, epochs + 1):
            with tqdm(total=iterations_per_epoch, desc='Training epoch {}'.format(ep)) as p_bar:
                for it in range(1, iterations_per_epoch + 1):
                    
                    # Perform one training step and obtain current loss value
                    loss = self._train_step(batch_size, **kwargs)

                    # Store returned loss
                    self.loss_history.add_entry(ep, loss)

                    # Compute running loss
                    avg_dict = self.loss_history.get_running_losses(ep)

                    # Get slope of loss trajectory for optional stopping
                    if self.lr_adjuster is not None:
                        slope = self.lr_adjuster.get_slope(self.loss_history.total_loss)
                    else:
                        slope = None

                    # Format for display on progress bar
                    disp_str = format_loss_string(ep, it, loss, avg_dict, slope)

                    # Update progress bar
                    p_bar.set_postfix_str(disp_str)
                    p_bar.update(1)

                    # Check optional stopping and end training
                    if self._check_optional_stopping():
                        self._save_trainer(save_checkpoint)
                        return self.loss_history.get_plottable()

            # Store after each epoch, if specified
            self._save_trainer(save_checkpoint)
        
        # self.loss_history.load_from_file(file_path=self.checkpoint_path)
        return self.loss_history.get_plottable()
    
    def train_offline(self, simulations_dict, epochs, batch_size, save_checkpoint=True,**kwargs):
        """ Trains an amortizer via offline learning. Assume parameters, data and optional 
        context have already been simulated (i.e., forward inference has been performed).

        Parameters
        ----------
        simulations_dict : dict
            A dictionaty containing the simulated data / context, if using the default keys, 
            the method expects at least the mandatory keys `sim_data` and `prior_draws` to be present
        epochs           : int
            Number of epochs (and number of times a checkpoint is stored)
        batch_size       : int
            Number of simulations to perform at each backpropagation step
        save_checkpoint  : bool (default - True)
            Determines whether to save checkpoints after each epoch,
            if a checkpoint_path provided during initialization, otherwise ignored.

        Returns
        -------
        losses : dict(ep_num : list(losses))
            A dictionary storing the losses across epochs and iterations
        Important
        ---------

        Examples
        --------
        # TODO
        """

        # Convert to custom data set
        data_set = SimulationDataset(simulations_dict, batch_size)

        self.loss_history.start_new_run()
        for ep in range(1, epochs + 1):

            with tqdm(total=int(np.ceil(data_set.n_sim / batch_size)), desc='Training epoch {}'.format(ep)) as p_bar:
                # Loop through dataset
                for bi, forward_dict in enumerate(data_set, start=1):

                    # Perform one training step and obtain current loss value
                    input_dict = self.configurator(forward_dict)
                    loss = self._train_step(batch_size, input_dict, **kwargs)

                    # Store returned loss
                    self.loss_history.add_entry(ep, loss)

                    # Compute running loss
                    avg_dict = self.loss_history.get_running_losses(ep)

                    # Get slope of loss trajectory for optional stopping
                    if self.lr_adjuster is not None:
                        slope = self.lr_adjuster.get_slope(self.loss_history.total_loss)
                    else:
                        slope = None

                    # Format for display on progress bar
                    disp_str = format_loss_string(ep, bi, loss, avg_dict, slope, it_str='Batch')

                    # Update progress
                    p_bar.set_postfix_str(disp_str)
                    p_bar.update(1)

                    # Check optional stopping and end training
                    if self._check_optional_stopping():
                        self._save_trainer(save_checkpoint)
                        return self.loss_history.get_plottable()

            # Store after each epoch, if specified
            if self.manager is not None and save_checkpoint:
                self._save_trainer(save_checkpoint)
        
        return self.loss_history.get_plottable()

    def train_from_presimulation(self, presimulation_path, max_epochs=None, save_checkpoint=True, custom_loader=None, **kwargs):

        """ Trains an amortizer via a modified form of offline training. 

        Like regular offline training, it assumes that parameters, data and optional context have already
        been simulated (i.e., forward inference has been performed).

        Also like regular offline training, it is faster than online training in scenarios where simulations are slow.
        Unlike regular offline training, it uses each batch from the presimulated dataset only once during training.
        A larger presimulated dataset is therefore required than for offline training, and the increase in speed
        gained by loading simulations instead of generating them on the fly comes at a cost: 
        a large presimulated dataset takes up a large amount of hard drive space.

        Parameters
        ----------
        presimulation_path : str
            File path to the folder containing the files from the precomputed simulation.
            Ideally generated using a GenerativeModel's presimulate_and_save method, otherwise must match
            the structure produced by that method: 

            Each file contains the data for one epoch, i.e. a number of batches, and must be compatible with the custom_loader provided:
            the custom_loader must read each file into a collection (either a dictionary or a list) of simulation_dict objects.
            This is easily achieved with the pickle library: if the files were generated from collections of simulation_dict objects
            using pickle.dump, the _default_loader (default for custom_load) will load them using pickle.load. 

            Training parameters like number of iterations and batch size are inferred from the files during training.
        save_checkpoint  : bool, optional, (default - True)
            Determines whether to save checkpoints after each epoch,
            if a checkpoint_path provided during initialization, otherwise ignored.

        custom_loader    : callable, optional, (default - _default_loader)
            Must take a string file_path as an input and output a collection (dictionary or list) of simulation_dict objects.
            A simulation_dict has the keys 'prior_non_batchable_context', 'prior_batchable_context', 'prior_draws', 'sim_non_batchable_context',
            'sim_batchable_context' and 'sim_data'. Unused keys should be paired with the value None; 'prior_draws' and 'sim_data' must have
            actual data as values.

        Returns
        -------
        losses : dict(ep_num : list(losses))
            A dictionary storing the losses across epochs and iterations
        """   
        
        # Use default loading function if none is provided
        if custom_loader is None:
            custom_loader = self._default_loader

        self.loss_history.start_new_run()

        # Loop over the presimulated dataset.
        file_list = os.listdir(presimulation_path)
        
        # Limit number of epochs to max_epochs
        if len(file_list) > max_epochs:
            file_list = file_list[:max_epochs]

        for ep, current_filename in enumerate(file_list, start=1):

            # Read single file into memory as a dictionary or list
            file_path = presimulation_path + '/' + current_filename
            epoch_data = custom_loader(file_path)
            
            # For each epoch, the number of iterations is inferred from the presimulated dictionary or list used for that epoch
            if isinstance(epoch_data, dict): 
                index_list = list(epoch_data.keys())
            elif isinstance(epoch_data, list):
                index_list = np.arange(len(epoch_data))    
            else:
                raise ValueError(f"Loading a simulation file resulted in a {type(epoch_data)}. Must be a dictionary or a list.")

            with tqdm(total=len(index_list), desc=f'Training epoch {ep}') as p_bar:
                for it, index in enumerate(index_list, start=1):

                    # Perform one training step and obtain current loss value
                    input_dict = self.configurator(epoch_data[index])

                    # Like the number of iterations, the batch size is inferred from presimulated dictionary or list
                    batch_size = len(input_dict[DEFAULT_KEYS['parameters']][0])
                    loss = self._train_step(batch_size, input_dict, **kwargs)

                    # Store returned loss
                    self.loss_history.add_entry(ep, loss)

                    # Compute running loss
                    avg_dict = self.loss_history.get_running_losses(ep)

                    # Get slope of loss trajectory for optional stopping
                    if self.lr_adjuster is not None:
                        slope = self.lr_adjuster.get_slope(self.loss_history.total_loss)
                    else:
                        slope = None

                    # Format for display on progress bar
                    disp_str = format_loss_string(ep, it, loss, avg_dict, slope)

                    # Update progress bar
                    p_bar.set_postfix_str(disp_str)
                    p_bar.update(1)

                    # Check optional stopping and end training
                    if self._check_optional_stopping():
                        self._save_trainer(save_checkpoint)
                        return self.loss_history.get_plottable()

            # Store after each epoch, if specified
            self._save_trainer(save_checkpoint)

        return self.loss_history.get_plottable()
    
    def train_rounds(self, rounds, sim_per_round, epochs, batch_size, save_checkpoint=True, **kwargs):
        """Trains an amortizer via round-based learning.

        Parameters
        ----------
        rounds          : int
            Number of rounds to perform (outer loop)
        sim_per_round   : int
            Number of simulations per round.
        epochs          : int
            Number of epochs (and number of times a checkpoint is stored, inner loop) within a round.
        batch_size      : int
            Number of simulations to use at each backpropagation step
        save_checkpoint : bool, optional, (default - True)
            A flag to decide whether to save checkpoints after each epoch,
            if a checkpoint_path provided during initialization, otherwise ignored

        Returns
        -------
        losses : dict(ep_num : list(losses))
            A dictionary storing the losses across epochs and iterations
        """

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        first_round = True

        for r in range(1, rounds + 1):
            # Data generation step
            if first_round:
                # Simulate initial data
                logger.info(f'Simulating initial {sim_per_round} data sets...')
                simulations_dict = self._forward_inference(sim_per_round, configure=False, **kwargs)
                first_round = False
            else:
                # Simulate further data
                logger.info(f'Simulating new {sim_per_round} data sets and appending to previous...')
                logger.info(f'New total number of simulated data sets: {sim_per_round * r}')
                simulations_dict_r = self._forward_inference(sim_per_round, configure=False, **kwargs)

                # Attempt to concatenate data sets
                for k in simulations_dict.keys():
                    if simulations_dict[k] is not None:
                        simulations_dict[k] = np.concatenate((simulations_dict[k], simulations_dict_r[k]), axis=0)
        
            # Train offline with generated stuff
            _ = self.train_offline(simulations_dict, epochs, batch_size, save_checkpoint, **kwargs)
        return self.loss_history.get_plottable()

    def _save_trainer(self, save_checkpoint):
        if self.manager is not None and save_checkpoint:
            self.manager.save()
            self.loss_history.save_to_file(file_path=self.checkpoint_path, max_to_keep=self.max_to_keep)
            if self.lr_adjuster is not None:
                self.lr_adjuster.save_to_file(file_path=self.checkpoint_path)
            if self.simulation_memory is not None:
                self.simulation_memory.save_to_file(file_path=self.checkpoint_path)

    def _check_optional_stopping(self):
        """ Helper method for checking optional stopping. Resets the adjuster
        if a stopping recommendation is issued. 
        """

        if self.lr_adjuster is None:
            return False
        if self.lr_adjuster.stopping_issued:
            self.lr_adjuster.reset()
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            logger.info('Optional stopping triggered.')
            return True
        return False

    def _train_step(self, batch_size, input_dict=None, **kwargs):
        """ Performs forward inference -> configuration -> network -> loss pipeline.

        Parameters
        ----------

        batch_size    : int 
            Number of simulations to perform at each backprop step
        input_dict    : dict
            The optional pre-configured forward dict from a generative model, simulated, if None
        **kwargs      : dict (default - {})
            Optional keyword arguments, which can be one of:
            `model_args` - optional keyword arguments passed to the generative model
            `conf_args`  - optional keyword arguments passed to the configurator
            `net_args`   - optional keyword arguments passed to the amortizer
        
        """

        if input_dict is None:
            input_dict = self._forward_inference(batch_size, **kwargs.pop('conf_args', {}), **kwargs.pop('model_args', {}))
        if self.simulation_memory is not None:
            self.simulation_memory.store(input_dict)
        loss = self._backprop_step(input_dict, **kwargs.pop('net_args', {}))
        return loss

    def _forward_inference(self, n_sim, configure=True, **kwargs):
        """ Performs one step of single-model forward inference.

        Parameters
        ----------
        n_sim         : int
            Number of simulations to perform at the given step (i.e., batch size)
        configure     : bool (default - True)
            Determines whether to pass the forward inputs through a configurator. 
        **kwargs      : dict
            Optional keyword arguments passed to the generative model

        Returns
        -------
        out_dict : dict
            The outputs of the generative model.

        Raises
        ------
        SimulationError
            If the trainer has no generative model but `trainer._forward_inference`
            is called (i.e., needs to simulate data from the generative model)
        """

        if self.generative_model is None:
            raise SimulationError("No generative model specified. Only offline learning is available!")
        out_dict = self.generative_model(n_sim, **kwargs.pop('model_args', {}))
        if configure:
            out_dict = self.configurator(out_dict, **kwargs.pop('conf_args', {}))
        return out_dict

    @tf.function
    def _backprop_step(self, input_dict, **kwargs):
        """ Computes the loss of the provided amortizer given an input dictionary and applies gradients.

         Parameters
        ----------
        input_dict  : dict
            The configured output of the genrative model
        **kwargs    : dict
            Optional keyword arguments passed to the network's compute_loss method
            
        Returns
        -------
        loss : dict
            The outputs of the compute_loss() method of the amortizer comprising all
            loss components, such as divergences or regularization.
        """

        # Forward pass and loss computation
        with tf.GradientTape() as tape:
            # Compute custom loss
            loss = self.amortizer.compute_loss(input_dict, training=True, **kwargs)
            # If dict, add components
            if type(loss) is dict:
                _loss = tf.add_n(list(loss.values()))
            else:
                _loss = loss
            # Collect regularization loss, if any
            if self.amortizer.losses != []:
                reg = tf.add_n(self.amortizer.losses)
                _loss += reg
                if type(loss) is dict:
                    loss['Regularization'] = reg
                else:
                    loss = {'Loss': loss, 'Regularization': reg}
        # One step backprop
        gradients = tape.gradient(_loss, self.amortizer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.amortizer.trainable_variables))

        return loss

    def _manage_configurator(self, config_fun, **kwargs):
        """ Determines which configurator to use if None specified during construction.      
        """

        # Do nothing if callable provided
        if callable(config_fun):
            return config_fun
        # If None (default), infer default config based on amortizer type
        else:
            # Amortized posterior
            if type(self.amortizer) is AmortizedPosterior:
                default_config = DefaultPosteriorConfigurator
                default_combiner = DefaultPosteriorCombiner()
                default_transfomer = DefaultPosteriorTransformer()

            # Amortized lieklihood
            elif type(self.amortizer) is AmortizedLikelihood:
                default_config = DefaultLikelihoodConfigurator
                default_combiner = DefaultLikelihoodTransformer()
                default_transfomer = DefaultLikelihoodCombiner()

            # Joint amortizer
            elif type(self.amortizer) is JointAmortizer:
                default_config = DefaultJointConfigurator
                default_combiner = DefaultJointTransformer()
                default_transfomer = DefaultJointCombiner()

            # Model comparison amortizer
            elif type(self.amortizer) is ModelComparisonAmortizer:
                if kwargs.get('n_models') is None:
                    raise ConfigurationError('Either your generative model or amortizer should have "n_models" attribute, or ' + 
                                             'you need initialize Trainer with n_models explicitly!')
                default_config = DefaultModelComparisonConfigurator(kwargs.get('n_models'))
            # Unknown raises an error
            else:
                raise NotImplementedError(f"Could not initialize configurator based on " +
                                          f"amortizer type {type(self.amortizer)}!")

        # Check string types
        # TODO: Make sure this works for all amortizers
        if type(config_fun) is str:
            if config_fun == "variable_num_obs":
                return default_config(
                    transform_fun=VariableObservationsTransformer(),
                    combine_fun=default_combiner)

            elif config_fun == 'one_hot':
                return default_config(
                    transform_fun=OneHotTransformer(),
                    combine_fun=default_combiner)

            elif config_fun == 'variable_num_obs_one_hot':
                return default_config(
                    transform_fun=TransformerUnion([
                        VariableObservationsTransformer(),
                        OneHotTransformer(),
                    ]),
                    combine_fun=default_combiner)
            elif config_fun == 'one_hot_variable_num_obs':
                return default_config(
                    transform_fun=TransformerUnion([
                        OneHotTransformer(),
                        VariableObservationsTransformer(),
                    ]),
                    combine_fun=default_combiner)
            else:
                raise NotImplementedError(f"Could not initialize configurator based on string" +
                                          f"argument should be in {STRING_CONFIGS}")

        elif config_fun is None:
            if type(self.amortizer) is ModelComparisonAmortizer:
                config_fun = default_config
            else:
                config_fun = default_config(
                    transform_fun=default_transfomer,
                    combine_fun=default_combiner
                )
            return config_fun
        else:
            raise NotImplementedError(f"Could not infer configurator based on provided type: {type(config_fun)}")
    
    def _check_consistency(self):
        """ Attempts to run one step generative_model -> configurator -> amortizer -> loss with 
        batch_size=2. Should be skipped if generative model has non-standard behavior.

        Raises
        ------
        ConfigurationError 
            If any operation along the above chain fails.
        """

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if self.generative_model is not None:
            _n_sim = 2
            try: 
                logger.info('Performing a consistency check with provided components...')
                _ = self.amortizer.compute_loss(self.configurator(self.generative_model(_n_sim)))
                logger.info('Done.')
            except Exception as err:
                raise ConfigurationError("Could not carry out computations of generative_model ->" +
                                         f"configurator -> amortizer -> loss! Error trace:\n {err}")
    
    def _default_loader(self, file_path):
        with open(file_path, 'rb+') as f:
                loaded_file = pickle_load(f)
        return(loaded_file)
