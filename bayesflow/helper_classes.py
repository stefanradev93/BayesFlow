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
import tensorflow as tf

from bayesflow.default_settings import DEFAULT_KEYS

class SimulatedDataset:
    """ Helper class to create a tensorflow Dataset which returns
    dictionaries in BayesFlow format.
    """
    
    def __init__(self, forward_dict, batch_size):
        """
        Create a tensorfow Dataset from forward inference outputs and determines format. 
        """
        
        slices, keys_used, keys_none, n_sim = self._determine_slices(forward_dict)
        self.data = tf.data.Dataset\
                .from_tensor_slices(tuple(slices))\
                .shuffle(self.n_sim)\
                .batch(batch_size)
        self.keys_used = keys_used
        self.keys_none = keys_none
        self.n_sim = n_sim
        
    def _determine_slices(self, forward_dict):
        """ Determine slices for a tensorflow Dataset.
        """
        
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
        """ Convert output of tensorflow Dataset to dict.
        """
        
        forward_dict = {}
        for key_used, batch_stuff in zip(self.keys_used, batch_in):
            forward_dict[key_used] = batch_stuff.numpy()
        for key_none in zip(self.keys_none):
            forward_dict[key_none] = None
        return forward_dict
    
    def __iter__(self):
        return map(self, self.data)


class ReduceLROnPlateau:
    """Reduce learning rate when a loss has stopped improving. Code inspired by:

    https://github.com/keras-team/keras/blob/v2.8.0/keras/callbacks.py#L2641-L2763

    Parameters
    ----------

    factor   : factor by which the learning rate will be reduced.
        `new_lr = lr * factor`.
    patience : number of epochs with no improvement after which learning rate
        will be reduced.
    verbose  : int. 0: quiet, 1: update messages.
    min_delta: threshold for measuring the new optimum, to only focus on
        significant changes.
    min_lr   : lower bound on the learning rate.

    """

    def __init__(self, factor=0.1, patience=3, min_delta=0.1, min_lr=0,):

        if factor >= 1.0:
            raise ValueError(f'ReduceLROnPlateau does not support a factor >= 1.0. Got {factor}')

        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best = 0
        self._reset()


    def _reset(self):
        """Resets wait counter and cooldown counter."""

        self.check_if_plateau = lambda a, b: np.less(a, b - self.min_delta)
        self.best = np.Inf
        self.wait = 0


    def on_epoch_end(self, history, optimizer, logs=None):

        num_epochs = list()


        # if self.monitor_op(current, self.best):
        # self.best = current
        # self.wait = 0
        # elif not self.in_cooldown():
        # self.wait += 1
        # if self.wait >= self.patience:
        # old_lr = self.model.optimizer.lr.numpy()
        # if old_lr > np.float32(self.min_lr):
        # new_lr = old_lr * self.factor
        # new_lr = max(new_lr, self.min_lr)
        # backend.set_value(self.model.optimizer.lr, new_lr)
        # self.wait = 0
