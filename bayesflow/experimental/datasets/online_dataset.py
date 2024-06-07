
import keras

from bayesflow.experimental.simulation import JointDistribution


class OnlineDataset(keras.utils.PyDataset):
    """
    A dataset that is generated on-the-fly.
    """
    def __init__(self, distribution, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.distribution = distribution
        self.batch_size = batch_size

    def __getitem__(self, item: int) -> (dict, dict):
        """ Sample a batch of data from the joint distribution unconditionally """
        return self.distribution.sample((self.batch_size,))

    @property
    def num_batches(self):
        # infinite dataset
        return None
