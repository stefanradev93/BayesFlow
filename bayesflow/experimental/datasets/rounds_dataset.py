
import keras

from bayesflow.experimental.simulation.distributions.generative_model import GenerativeModel


class RoundsDataset(keras.utils.PyDataset):
    """
    A dataset that is generated on-the-fly at the beginning of every n-th epoch.
    """
    def __init__(self, generative_model: GenerativeModel, batch_size: int, batches_per_epoch: int, epochs_per_round: int, **kwargs):
        super().__init__(**kwargs)
        self.generative_model = generative_model
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.epochs_per_round = epochs_per_round
        self.epoch = 0

        self.data = None

        self.regenerate()

    def __getitem__(self, item: int) -> tuple:
        """ Get a batch of pre-simulated data """
        data = self.data[item]
        return (data,)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def on_epoch_end(self) -> None:
        self.epoch += 1
        if self.epoch % self.epochs_per_round == 0:
            self.regenerate()

    def regenerate(self) -> None:
        """ Sample batches of data from the generative model """
        self.data = [self.generative_model.sample((self.batch_size,)) for _ in range(self.batches_per_epoch)]
