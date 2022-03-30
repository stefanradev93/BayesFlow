# Copyright 2022 The BayesFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from tqdm.notebook import tqdm

import logging

from bayesflow.forward_inference import MultiGenerativeModel
logging.basicConfig()

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from bayesflow.configuration import *
from bayesflow.exceptions import SimulationError
from bayesflow.helper_functions import apply_gradients, format_loss_string
from bayesflow.helper_classes import SimulationDataset, LossHistory, SimulationMemory
from bayesflow.default_settings import STRING_CONFIGS
from bayesflow.amortized_inference import *


class Trainer:
    """ This class connects a generative model (or, already simulated data from a model) with
    a configurator and a neural inference architecture for amortized inference (amortizer). A Trainer 
    is responsible for optimizing the amortizer via various forms of simulation-based training.

    At the very minium, the trainer must be initialized with an `amortizer` instance, which is capable
    of processing the (configured) outputs of a generative model. A `configurator` will then process
    the outputs of the generative model and convert them in suitable inputs for the amortizer. Users
    can choose from a palette of default configurators or create their own configurators, essentially
    building a modularized pipeline GenerativeModel -> Configurator -> Amortizer.

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
    
    Note: For extremely slow simulators (i.e., more than an hour of a single simulation), the BayesFlow framework might not be the ideal
    choice and should be considered in combination with a black-box surrogate optimization method, such as Bayesian optimization.
    """

    def __init__(self, amortizer, generative_model=None, configurator=None, optimizer=None,
                 learning_rate=0.0005, checkpoint_path=None, max_to_keep=5, clip_method='global_norm', 
                 clip_value=None, skip_checks=False, **kwargs):
        """ Creates a trainer which will use a generative model (or data simulated from it) to optimize
        a neural arhcitecture (amortizer) for amortized posterior inference, likelihood inference, or both.

        Parameters
        ----------
        amortizer        : bayesflow.amortizers.Amortizer
            The neural architecture to be optimized
        generative_model : bayesflow.forward_inference.GenerativeModel
            A generative model returning a dictionary with randomly sampled parameters, data, and optional context
        configurator     : callable 
            A callable object transforming and combining the outputs of the generative model into inputs for BayesFlow
        optimizer        : tf.keras.optimizer.Optimizer or None
            Optimizer for the neural network. ``None`` will result in `tf.keras.optimizers.Adam`
        learning_rate    : float or tf.keras.schedules.LearningRateSchedule
            The learning rate used for the optimizer
        checkpoint_path  : string, optional
            Optional folder name for storing the trained network
        max_to_keep      : int, optional
            Number of checkpoints to keep
        clip_method      : {'norm', 'value', 'global_norm'}
            Optional gradient clipping method
        clip_value       : float
            The value used for gradient clipping when clip_method is in {'value', 'norm'}
        skip_checks      : boolean
            If True, do not perform consistency checks, i.e., simulator runs and passed through nets
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

        # Gradient clipping settings
        self.clip_method = clip_method
        self.clip_value = clip_value
        
        # Optimizer settings
        if optimizer is None:
            self.optimizer = Adam(learning_rate)
        else:
            self.optimizer = optimizer(learning_rate)

        # Checkpoint settings
        if checkpoint_path is not None:
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.amortizer)
            self.manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=max_to_keep)
            self.checkpoint.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                logger.info("Networks loaded from {}".format(self.manager.latest_checkpoint))
            else:
                logger.info("Initialized networks from scratch.")
        else:
            self.checkpoint = None
            self.manager = None
        self.checkpoint_path = checkpoint_path

        # Set-up memory classes
        self.loss_history= LossHistory()
        self.simulation_memory = SimulationMemory()

        # Perform a sanity check wiuth provided components
        if not skip_checks:
            self._check_consistency()

    def load_pretrained_network(self):
        """Attempts to load a pre-trained network if checkpoint path is provided and a checkpoint manager exists.
        """

        if self.manager is None or self.checkpoint is None:
            return False
        status = self.checkpoint.restore(self.manager.latest_checkpoint)
        return status

    def train_online(self, epochs, iterations_per_epoch, batch_size, save_checkpoint=True, **kwargs):
        """Trains an amortizer via online learning. Additional keyword arguments
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

                    # Format for display on progress bar
                    disp_str = format_loss_string(ep, it, loss, avg_dict)

                    # Update progress bar
                    p_bar.set_postfix_str(disp_str)
                    p_bar.update(1)

            # Store after each epoch, if specified
            if self.manager is not None and save_checkpoint:
                self.manager.save()
        return self.loss_history.get_copy()

    def train_offline(self, simulations_dict, epochs, batch_size, save_checkpoint=True,**kwargs):
        """ Trains an amortizer via offline learning. Assume parameters, data and optional 
        context have already been simulated (i.e., forward inference has been performed).

        Parameters
        ----------
        simulations_dict : dict
            A dictionaty containing the simulated data / context, if using the default keys, 
            the method expects mandatory keys `sim_data` and `prior_draws` to be present
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
                for bi, forward_dict in enumerate(data_set):

                    # Perform one training step and obtain current loss value
                    input_dict = self.configurator(forward_dict)
                    loss = self._train_step(batch_size, input_dict, **kwargs)

                    # Store returned loss
                    self.loss_history.add_entry(ep, loss)

                    # Compute running loss
                    avg_dict = self.loss_history.get_running_losses(ep)

                    # Format for display on progress bar
                    disp_str = format_loss_string(ep, bi, loss, avg_dict)

                    p_bar.set_postfix_str(disp_str)
                    p_bar.update(1)

            # Store after each epoch, if specified
            if self.manager is not None and save_checkpoint:
                self.manager.save()
        return self.loss_history.get_copy()

    def train_rounds(self, rounds, sim_per_round, epochs, batch_size, save_checkpoint=True, **kwargs):
        """Trains an amortizer via round-based learning.

        Parameters
        ----------
        rounds         : int
            Number of rounds to perform (outer loop)
        sim_per_round  : int
            Number of simulations per round.
        epochs         : int
            Number of epochs (and number of times a checkpoint is stored, inner loop) within a round.
        batch_size     : int
            Number of simulations to use at each backpropagation step
        save_checkpoint      : bool (default - True)
            A flag to decide whether to save checkpoints after each epoch,
            if a checkpoint_path provided during initialization, otherwise ignored

        Returns
        -------
        losses : dict(ep_num : list(losses))
            A dictionary storing the losses across epochs and iterations
        """

        logger = logging.getLogger()
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
        return self.loss_history.get_copy()

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
        loss = self._backprop_step(input_dict, **kwargs.pop('net_args', {}))
        return loss

    def _forward_inference(self, n_sim, configure=True, **kwargs):
        """
        Performs one step of single-model forward inference.

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

    def _backprop_step(self, input_dict, **kwargs):
        """Computes loss and applies gradients.

         Parameters
        ----------
        input_dict  : dict
            The configured output of the genrative model
        **kwargs    : dict
            Optional keyword arguments passed to the network's compute_loss method
            
        Returns
        -------
        out_dict : dict
            The outputs of the generative model.
        """

        # Forward pass and loss computation
        with tf.GradientTape() as tape:
            # Compute custom loss
            loss = self.amortizer.compute_loss(input_dict, **kwargs)
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
        apply_gradients(self.optimizer, gradients, self.amortizer.trainable_variables, 
                        self.clip_value, self.clip_method)

        return loss

    def _manage_configurator(self, config_fun, **kwargs):
        """ Determines which configurator to use if None specified.        
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
        """Attempts to run one step generative_model -> configurator -> amortizer -> loss."""

        logger = logging.getLogger()
        if self.generative_model is not None:
            _n_sim = 2
            try: 
                logger.info('Performing a consistency check with provided components...')
                _ = self.amortizer.compute_loss(self.configurator(self.generative_model(_n_sim)))
                logger.info('Done.')
            except Exception as err:
                raise ConfigurationError("Could not carry out computations of generative_model ->" +
                                         f"configurator -> amortizer -> loss! Error trace:\n {err}")