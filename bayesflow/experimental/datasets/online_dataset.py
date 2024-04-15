
import keras

from bayesflow.experimental.simulation.distributions.generative_model import GenerativeModel


class OnlineDataset(keras.utils.PyDataset):
    """
    A dataset that is generated on-the-fly.
    """
    def __init__(self, generative_model: GenerativeModel, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.generative_model = generative_model
        self.batch_size = batch_size

    def __getitem__(self, item: int) -> (dict,):
        """ Sample a batch of data from the generative model """
        data = self.generative_model.sample((self.batch_size,))
        return (data,)

    def __len__(self) -> int:
        # signals infinite dataset
        return -1
