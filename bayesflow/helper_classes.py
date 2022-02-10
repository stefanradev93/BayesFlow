import tensorflow as tf


class SimulatedDataset:
    """ Helper class to create a tensorflow Dataset which returns
    dictionaries in BayesFlow format.
    """
    
    def __init__(self, forward_dict, batch_size):
        """
        Create a tensorfow Dataset from forward inference outputs and determines format. 
        """
        
        slices, keys_used, keys_none = self._determine_slices(forward_dict)
        self.keys_used = keys_used
        self.keys_none = keys_none
        self.data = tf.data.Dataset\
                .from_tensor_slices(tuple(slices))\
                .shuffle(self.n_sim)\
                .batch(batch_size)
        
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
        self.n_sim = forward_dict['sim_data'].shape[0]
        return slices, keys_used, keys_none
    
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