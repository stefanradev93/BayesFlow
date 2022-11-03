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

from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import re

try:
    import cPickle as pickle
except:
    import pickle

import logging
logging.basicConfig()

from sklearn.linear_model import HuberRegressor

from bayesflow.default_settings import DEFAULT_KEYS


class SimulationDataset:
    """Helper class to create a tensorflow.data.Dataset which parses simulation dictionaries 
    and returns simulation dictionaries as expected by BayesFlow amortizers.
    """
    
    def __init__(self, forward_dict, batch_size):
        """Creates a tensorfow.data.Dataset from forward inference outputs and determines format."""
        
        slices, keys_used, keys_none, n_sim = self._determine_slices(forward_dict)
        self.data = tf.data.Dataset\
                .from_tensor_slices(tuple(slices))\
                .shuffle(n_sim)\
                .batch(batch_size)
        self.keys_used = keys_used
        self.keys_none = keys_none
        self.n_sim = n_sim
        
    def _determine_slices(self, forward_dict):
        """Determine slices for a tensorflow Dataset."""
        
        keys_used = []
        keys_none = []
        slices = []
        for k, v in forward_dict.items():
            if forward_dict[k] is not None:
                slices.append(v)
                keys_used.append(k)
            else:
                keys_none.append(k)
        n_sim = forward_dict[DEFAULT_KEYS['sim_data']].shape[0]
        return slices, keys_used, keys_none, n_sim
    
    def __call__(self, batch_in):
        """Convert output of tensorflow.data.Dataset to dict."""
        
        forward_dict = {}
        for key_used, batch_stuff in zip(self.keys_used, batch_in):
            forward_dict[key_used] = batch_stuff.numpy()
        for key_none in zip(self.keys_none):
            forward_dict[key_none] = None
        return forward_dict
    
    def __iter__(self):
        return map(self, self.data)


class RegressionLRAdjuster:
    """This class will compute the slope of the loss trajectory and inform learning rate decay."""
    
    file_name = 'lr_adjuster'
    
    def __init__(self, optimizer, period=1000, wait_between_fits=10, patience=10, tolerance=-0.05, 
                 reduction_factor=0.25, cooldown_factor=2, num_resets=3, **kwargs):
        """Creates an instance with given hyperparameters which will track the slope of the 
        loss trajectory according to specified hyperparameters and then issue an optional
        stopping suggestion.
        
        Parameters
        ----------

        optimizer         : tf.keras.optimizers.Optimizer instance
            An optimizer implementing a lr() method
        period            : int, optional, default: 1000
            How much loss values to consider from the past
        wait_between_fits : int, optional, default: 10
            How many backpropagation updates to wait between two successive fits
        patience          : int, optional, default: 10
            How many successive times the tolerance value is reached before lr update.
        tolerance         : float, optional, default: -0.05
            The minimum slope to be considered substantial for training.
        reduction_factor  : float in [0, 1], optional, default: 0.25
            The factor by which the learning rate is reduced upon hitting the `tolerance`
            threshold for `patience` number of times
        cooldown_factor   : float, optional, default: 2
            The factor by which the `period` is multiplied to arrive at a cooldown period.
        num_resets        : int, optional, default: 3
            How many times to reduce the learning rate before issuing an optional stopping
        **kwargs          : dict, optional, default {}
            Additional keyword arguments passed to the `HuberRegression` class.
        """
        
        self.optimizer = optimizer
        self.period = period
        self.wait_between_periods = wait_between_fits
        self.regressor = HuberRegressor(**kwargs)
        self.t_vector = np.linspace(0, 1, self.period)[:, np.newaxis]
        self.patience = patience
        self.tolerance = tolerance
        self.num_resets = num_resets
        self.reduction_factor = reduction_factor
        self.stopping_issued = False
        self.cooldown_factor = cooldown_factor
        self._history = {
            'iteration'    : [], 
            'learning_rate': []
        }
        self._reset_counter = 0
        self._patience_counter = 0
        self._cooldown_counter = 0
        self._wait_counter = 0
        self._slope = None
        self._is_waiting = False
        self._in_cooldown = False
        
    def get_slope(self, losses):
        """Fits a Huber regression on the provided loss trajectory or returns `None` if
        not enough data points present.
        """
        
        # Return None if not enough loss values present
        if losses.shape[0] < self.period:
            return None

        # Increment counter
        if self._in_cooldown:
            self._cooldown_counter += 1
        
        # Check if still in a waiting phase and return old slope
        # if still waiting, otherwise refit Huber regression
        wait = self._check_waiting()
        if wait:
            return self._slope
        else:
            self.regressor.fit(self.t_vector, losses[-self.period:])
            self._slope = self.regressor.coef_[0]
            self._check_patience()
            return self._slope

    def reset(self):
        """Resets all stateful variables in preparation for a new start."""

        self._reset_counter = 0
        self._patience_counter = 0
        self._cooldown_counter = 0
        self._wait_counter = 0
        self._in_cooldown = False
        self._is_waiting = False
        self.stopping_issued = False

    def save_to_file(self, file_path):
        """Saves the state parameters of a RegressionLRAdjuster object to a pickled dictionary in file_path."""

        # Create path to memory
        memory_path = os.path.join(file_path, f'{RegressionLRAdjuster.file_name}.pkl')
        
        # Prepare attributes
        states_dict = {}
        states_dict['_history'] = self._history
        states_dict['_reset_counter'] = self._reset_counter
        states_dict['_patience_counter'] = self._patience_counter
        states_dict['_cooldown_counter'] = self._cooldown_counter
        states_dict['_wait_counter'] = self._wait_counter
        states_dict['_slope'] = self._slope
        states_dict['_is_waiting'] = self._is_waiting
        states_dict['_in_cooldown'] = self._in_cooldown
        
        # Dump as pickle object
        with open(memory_path, 'wb') as f:
            pickle.dump(states_dict, f)

    def load_from_file(self, file_path):
        """Loads the saved LRAdjuster object from file_path."""

        # Logger init
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Create path to memory
        memory_path = os.path.join(file_path, f'{RegressionLRAdjuster.file_name}.pkl')

        # Case memory file exists
        if os.path.exists(memory_path):

            # Load pickle and fill in attributes
            with open(memory_path, 'rb') as f:
                states_dict = pickle.load(f)

            self._history = states_dict['_history']
            self._reset_counter = states_dict['_reset_counter']
            self._patience_counter = states_dict['_patience_counter']
            self._cooldown_counter = states_dict['_cooldown_counter']
            self._wait_counter = states_dict['_wait_counter']
            self._slope = states_dict['_slope']
            self._is_waiting = states_dict['_is_waiting']
            self._in_cooldown = states_dict['_in_cooldown']

            logger.info(f"Loaded RegressionLRAdjuster from {memory_path}")

        # Case memory file does not exist
        else:
            logger.info("Initialized a new RegressionLRAdjuster.")

    def _check_patience(self):
        """Determines whether to reduce learning rate or be patient."""

        # Do nothing, if still in cooldown period
        if self._in_cooldown and self._cooldown_counter < int(self.cooldown_factor * self.period):
            return 
        # Otherwise set cooldown flag to False and reset counter
        else:
            self._in_cooldown = False
            self._cooldown_counter = 0

        # Check if negetaive slope too small
        if self._slope > self.tolerance:
            self._patience_counter += 1
        else:
            self._patience_counter = max(0, self._patience_counter - 1)

        # Check if patience surpassed and issue a reduction in learning rate
        if self._patience_counter >= self.patience:
            self._reduce_learning_rate()
            self._patience_counter = 0

    def _reduce_learning_rate(self):
        """Reduces the learning rate by a given factor."""

        # Logger init
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        if self._reset_counter >= self.num_resets:
            self.stopping_issued = True
        else:
            # Take care of updating learning rate
            old_lr = self.optimizer.lr.numpy()
            new_lr = round(self.reduction_factor * old_lr, 8)
            self.optimizer.lr.assign(new_lr)
            self._reset_counter += 1

            # Store iteration and learning rate
            self._history['iteration'].append(self.optimizer.iterations.numpy())
            self._history['learning_rate'].append(old_lr)

            # Verbose info to user
            logger.info(f'Reducing learning rate from {old_lr:.8f} to: {new_lr:.8f} and entering cooldown...')

            # Set cooldown flag to avoid reset for some time given by self.period
            self._in_cooldown = True

    def _check_waiting(self):
        """Determines whether to compute a new slope or wait."""

        # Case currently waiting
        if self._is_waiting:
            # Case currently waiting but period is over
            if self._wait_counter >= self.wait_between_periods - 1:
                self._wait_counter = 0
                self._is_waiting = False
            # Case currently waiting and period not over
            else:
                self._wait_counter += 1
            return True
        # Case not waiting
        else:
            self._is_waiting = True
            self._wait_counter += 1
            return False


class LossHistory:
    """Helper class to keep track of losses during training."""

    file_name = 'history'

    def __init__(self):
        self.history = {}
        self.loss_names = []
        self._current_run = 0
        self._total_loss = []
        self.latest = 0

    @property
    def total_loss(self):
        return np.array(self._total_loss)

    def start_new_run(self):
        self._current_run += 1
        self.history[f'Run {self._current_run}'] = {}

    def add_entry(self, epoch, current_loss):
        """ Adds loss entry for current epoch into internal memory data structure.
        """

        # Add epoch key, if specified
        if self.history[f'Run {self._current_run}'].get(f'Epoch {epoch}') is None:
            self.history[f'Run {self._current_run}'][f'Epoch {epoch}'] = []

        # Handle dict loss output
        if type(current_loss) is dict:
            # Store keys, if none existing
            if self.loss_names == []:
                self.loss_names = [k for k in current_loss.keys()]
            
            # Create and store entry
            entry = [v.numpy() if type(v) is not np.ndarray else v for v in current_loss.values()]
            self.history[f'Run {self._current_run}'][f'Epoch {epoch}'].append(entry)

            # Add entry to total loss
            self._total_loss.append(sum(entry))

        # Handle tuple or list loss output
        elif type(current_loss) is tuple or type(current_loss) is list:
            entry = [v.numpy() if type(v) is not np.ndarray else v for v in current_loss]
            self.history[f'Run {self._current_run}'][f'Epoch {epoch}'].append(entry)
            # Store keys, if none existing
            if self.loss_names == []:
                self.loss_names = [f'Loss.{l}' for l in range(1, len(entry)+1)]

            # Add entry to total loss
            self._total_loss.append(sum(entry))
        
        # Assume scalar loss output
        else:
            self.history[f'Run {self._current_run}'][f'Epoch {epoch}'].append(current_loss.numpy())
            # Store keys, if none existing
            if self.loss_names == []:
                self.loss_names.append('Default.Loss')
            
            # Add entry to total loss
            self._total_loss.append(current_loss.numpy())

    def get_running_losses(self, epoch):
        """Compute and return running means of the losses for current epoch."""

        means = np.atleast_1d(np.mean(self.history[f'Run {self._current_run}'][f'Epoch {epoch}'], axis=0))
        if means.shape[0] == 1:    
            return {'Avg.Loss': means[0]}
        else:
            return {'Avg.' + k: v for k, v in zip(self.loss_names, means)}

    def get_plottable(self):
        """Returns the losses as a nicely formatted pandas DataFrame."""

        # Assume equal lengths per epoch and run
        try:
            losses_list = [pd.melt(pd.DataFrame.from_dict(self.history[r], orient='index').T) for r in self.history]
            losses_list = pd.concat(losses_list, axis=0).value.to_list()
            losses_list = [l for l in losses_list if l is not None]
            losses_df = pd.DataFrame(losses_list, columns=self.loss_names)
            return losses_df
        # Handle unequal lengths or problems when user kills training with an interrupt
        except ValueError as ve:
            return self.history
        except TypeError as te:
            return self.history

    def flush(self):
        """Returns current history and removes all existing loss history."""

        h = self.history
        self.history = {}
        self._current_run = 0
        return h

    def get_copy(self):
        return deepcopy(self.history)
    
    def save_to_file(self, file_path, max_to_keep):
        """Saves a LossHistory object to a pickled dictionary in file_path.
         If max_to_keep saved loss history files are found in file_path, the oldest is deleted before a new one is saved.
         """
        
        # Increment history index
        self.latest += 1
        
        # Path to history
        history_path = os.path.join(file_path, f'{LossHistory.file_name}_{self.latest}.pkl')
        
        # Prepare full history dict 
        full_history_dict = self.get_copy()
        full_history_dict['loss_names'] = self.loss_names
        full_history_dict['_current_run'] = self._current_run
        full_history_dict['_total_loss'] = self._total_loss
        
        # Pickle current
        with open(history_path, 'wb') as f:
            pickle.dump(full_history_dict, f)
        
        # Get list of history checkpoints
        history_checkpoints_list = [l for l in os.listdir(file_path) if 'history' in l]
            
        # Determine the oldest saved loss history and remove it
        if len(history_checkpoints_list) > max_to_keep:
            oldest_history_path = os.path.join(file_path, f'history_{self.latest-max_to_keep}.pkl')
            os.remove(oldest_history_path)
            
    def load_from_file(self, file_path):
        """Loads the most recent saved LossHistory object from file_path."""

        # Logger init
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Get list of histories
        if os.path.exists(file_path):
            history_checkpoints_list = [l for l in os.listdir(file_path) if LossHistory.file_name in l]
        else:
            history_checkpoints_list = []
        
        # Case history list is not empty
        if len(history_checkpoints_list) > 0:
            
            # Determine which file contains the latest LossHistory and load it
            file_numbers = [int(re.findall(r'\d+', h)[0]) for h in history_checkpoints_list]
            latest_file = history_checkpoints_list[np.argmax(file_numbers)]
            latest_number = np.max(file_numbers)
            latest_path = os.path.join(file_path, latest_file)
            
            # Load dictionary
            with open(latest_path, 'rb') as f:
                full_history_dict = pickle.load(f)
                
            # Fill entries
            self.latest = latest_number
            self._total_loss = full_history_dict['_total_loss']
            self._current_run = full_history_dict['_current_run']
            self.loss_names = full_history_dict['loss_names']
            self.history = {k:v for k, v in full_history_dict.items() if k not in ['_total_loss', '_current_run', 'loss_names']}
            
            # Verbose
            logger.info(f"Loaded loss history from {latest_path}")

        # Case history list is empty
        else:
            logger.info("Initialized empty loss history.")


class SimulationMemory:
    """Helper class to keep track of a pre-determined number of simulations during training."""

    file_name = 'memory'

    def __init__(self, stores_raw=True, capacity_in_batches=50):
        self.stores_raw = stores_raw
        self._capacity = capacity_in_batches
        self._buffer = [None] * self._capacity
        self._idx = 0
        self.size_in_batches = 0

    def store(self, forward_dict):
        """Stores simulation outputs in `forward_dict`, if internal buffer is not full.

        Parameters
        ----------
        forward_dict : dict
            The configured outputs of the forward model.
        """

        # If full, overwrite at index
        if not self.is_full():
            self._buffer[self._idx] = forward_dict
            self._idx += 1
            self.size_in_batches += 1
    
    def get_memory(self):
        return deepcopy(self._buffer)

    def is_full(self):
        """Returns True if the buffer is full, otherwise False."""

        if self._idx >= self._capacity:
            return True
        return False
    
    def save_to_file(self, file_path):
        """Saves a `SimulationMemory` object to a pickled dictionary in file_path."""

        # Create path to memory
        memory_path = os.path.join(file_path, f'{SimulationMemory.file_name}.pkl')
        
        # Prepare attributes
        full_memory_dict = {}
        full_memory_dict['stores_raw'] = self.stores_raw
        full_memory_dict['_capacity'] = self._capacity
        full_memory_dict['_buffer'] = self._buffer
        full_memory_dict['_idx'] = self._idx
        full_memory_dict['_size_in_batches'] = self.size_in_batches
        
        # Dump as pickle object
        with open(memory_path, 'wb') as f:
            pickle.dump(full_memory_dict, f)

    def load_from_file(self, file_path):
        """Loads the saved `SimulationMemory` object from file_path."""

        # Logger init
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Create path to memory
        memory_path = os.path.join(file_path, f'{SimulationMemory.file_name}.pkl')

        # Case memory file exists
        if os.path.exists(file_path):

            # Load pickle and fill in attributes
            with open(memory_path, 'rb') as f:
                full_memory_dict = pickle.load(f)

            self.stores_raw = full_memory_dict['stores_raw']
            self._capacity = full_memory_dict['_capacity']
            self._buffer = full_memory_dict['_buffer']
            self._idx = full_memory_dict['_idx']
            self.size_in_batches = full_memory_dict['_size_in_batches']
            logger.info(f"Loaded simulation memory from {memory_path}")

        # Case memory file does not exist
        else:
            logger.info("Initialized empty simulation memory.")


class MemoryReplayBuffer:
    """Implements a memory replay buffer for simulation-based inference.

    Attributes
    ----------
    TODO
    """

    def __init__(self, stores_raw=True, capacity_in_batches=50):
        """TODO"""

        self.stores_raw = stores_raw
        self._capacity = capacity_in_batches
        self._buffer = [None] * self._capacity
        self._idx = 0
        self._size_in_batches = 0
        self._is_full = False

    def store(self, forward_dict):
        """ Stores simulation outputs, if internal buffer is not full.

        Parameters
        ----------
        forward_dict : dict
            The (raw or configured) outputs of the forward model.
        """

        # If full, overwrite at index
        if self._is_full:
            self._overwrite(forward_dict)
        
        # Otherwise still capacity to append
        else:
            # Add to internal list
            self._buffer[self._idx] = forward_dict

            # Increment index and # of batches currently stored
            self._idx += 1
            self._size_in_batches += 1

            # Check whether buffer is full and set flag if thats the case
            if self._idx == self._capacity:
                self._is_full = True
            
    def sample(self):
        """ Samples `batch_size` number of parameter vectors and simulations from buffer.

        Returns
        -------
        forward_dict : dict
            The (raw or configured) outputs of the forward model.
        """
        
        rand_idx = np.random.default_rng().integers(low=0, high=self._size_in_batches)
        return self._buffer[rand_idx]

    def _overwrite(self, forward_dict):
        """Overwrites a simulated batch at current position. Only called when the internal buffer is full."""

        # Reset index, if at the end of buffer
        if self._idx == self._capacity:
            self._idx = 0

        # Overwrite params and data at index
        self._buffer[self._idx] = forward_dict

        # Increment index
        self._idx += 1
