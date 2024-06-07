
import keras

from bayesflow.experimental.simulation import JointDistribution


class RoundsDataset(keras.utils.PyDataset):
    """
    A dataset that is generated on-the-fly at the beginning of every n-th epoch.
    """
    def __init__(self, joint_distribution: JointDistribution, batch_size: int, batches_per_epoch: int, epochs_per_round: int, **kwargs):
        super().__init__(**kwargs)
        self.joint_distribution = joint_distribution
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
        self.data = [self.joint_distribution.sample((self.batch_size,)) for _ in range(self.batches_per_epoch)]
