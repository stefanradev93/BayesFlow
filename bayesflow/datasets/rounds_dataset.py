import keras

from bayesflow.configurators import Configurator
from bayesflow.simulators.simulator import Simulator
from bayesflow.types import Tensor


class RoundsDataset(keras.utils.PyDataset):
    """
    A dataset that is generated on-the-fly at the beginning of every n-th epoch.
    """

    def __init__(
        self,
        simulator: Simulator,
        batch_size: int,
        batches_per_epoch: int,
        epochs_per_round: int,
        configurator: Configurator,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if keras.backend.backend() == "torch" and kwargs.get("use_multiprocessing"):
            # keras workaround: https://github.com/keras-team/keras/issues/19346
            import multiprocessing as mp

            mp.set_start_method("spawn", force=True)

        self.batches = None
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.configurator = configurator
        self.epoch = 0
        self.epochs_per_round = epochs_per_round
        self.simulator = simulator

        self.regenerate()

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        """Get a batch of pre-simulated data"""
        batch = self.batches[item]
        if self.configurator is not None:
            batch = self.configurator.configure(batch)
        return batch

    @property
    def num_batches(self):
        # infinite dataset
        return None

    def on_epoch_end(self) -> None:
        self.epoch += 1
        if self.epoch % self.epochs_per_round == 0:
            self.regenerate()

    def regenerate(self) -> None:
        """Sample new batches of data from the joint distribution unconditionally"""
        self.batches = [self.simulator.sample((self.batch_size,)) for _ in range(self.batches_per_epoch)]
