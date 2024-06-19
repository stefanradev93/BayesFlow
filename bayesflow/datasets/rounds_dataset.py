
import keras

from bayesflow.simulators.simulator import Simulator


class RoundsDataset(keras.utils.PyDataset):
    """
    A dataset that is generated on-the-fly at the beginning of every n-th epoch.
    """
    def __init__(self, simulator: Simulator, batch_size: int, batches_per_epoch: int, epochs_per_round: int, **kwargs):
        super().__init__(**kwargs)

        if kwargs.get("use_multiprocessing"):
            # keras workaround: https://github.com/keras-team/keras/issues/19346
            import multiprocessing as mp
            mp.set_start_method("spawn", force=True)

        self.simulator = simulator
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.epochs_per_round = epochs_per_round
        self.epoch = 0

        self.data = None

        self.regenerate()

    def __getitem__(self, item: int) -> (dict, dict):
        """ Get a batch of pre-simulated data """
        return self.data[item]

    @property
    def num_batches(self):
        # infinite dataset
        return None

    def on_epoch_end(self) -> None:
        self.epoch += 1
        if self.epoch % self.epochs_per_round == 0:
            self.regenerate()

    def regenerate(self) -> None:
        """ Sample new batches of data from the joint distribution unconditionally """
        self.data = [self.simulator.sample((self.batch_size,)) for _ in range(self.batches_per_epoch)]
