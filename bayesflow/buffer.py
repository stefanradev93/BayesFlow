import numpy as np


class MemoryReplayBuffer:
    """Implements a memory replay buffer for simulation-based inference.

    Attributes
    ----------
    capacity: int
        Maximum number of batches to store in the buffer
    size_in_batches: int
        Number of currently stored batches
    _buffer : dict
        Buffer data as a dictionary with keys `'params', 'sim_data'`
    """

    def __init__(self, capacity):
        """Creates a memory replay buffer for simulation-based inference.

        Parameters
        ----------
        capacity : int
            Maximum number of batches to store in buffer
        """

        assert capacity >= 1, 'capacity should be a positive integer in (0, inf)'

        self.capacity = capacity
        self._buffer = {
            'params': [None] * capacity,
            'sim_data': [None] * capacity
        }
        self._idx = 0
        self.size_in_batches = 0
        self._is_full = False

    def store(self, params, sim_data):
        """ Stores params and simulated data.

        If buffer is not full, stores params and data at the end, if full, overwrites params and data at current index.

        Parameters
        ----------
        params: object
            Parameters to be stored
        sim_data: object
            Simulated data to be stored
        """

        # If full, overwrite at index
        if self._is_full:
            self._overwrite(params, sim_data)
        
        # Otherwise still capacity to append
        else:
            # Add to list
            self._buffer['params'][self._idx] = params
            self._buffer['sim_data'][self._idx] = sim_data

            # Increment index and # of batches currently stored
            self._idx += 1
            self.size_in_batches += 1

            # Check whether buffer is full and set flag if thats the case
            if self._idx == self.capacity:
                self._is_full = True
            
    def sample(self):
        """ Samples `batch_size` number of parameter vectors and simulations from buffer.

        Returns
        -------
        params    : np.array(np.float32)
            Array of sampled parameters of shape ``(batch_size, param_dim)``
        sim_data  : np.array(np.float32)
            Array of simulated data sets or summary statistics thereof,
            shape ``(batch_size, n_obs, data_dim)`` or ``(batch_size, sum_dim)``
        """
        
        rand_idx = np.random.randint(0, self.size_in_batches)
        params = self._buffer['params'][rand_idx]
        sim_data = self._buffer['sim_data'][rand_idx]
        return params, sim_data

    def _overwrite(self, params, sim_data):
        """Only called when the internal buffer is full. Overwrites params and sim_data at current index.
        """

        # Reset index, if at the end of buffer
        if self._idx == self.capacity:
            self._idx = 0

        # Overwrite params and data at index
        self._buffer['params'][self._idx] = params
        self._buffer['sim_data'][self._idx] = sim_data

        # Increment index
        self._idx += 1
