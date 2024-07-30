import keras
from optree import PyTree

from bayesflow.data_adapters import DataAdapter
from bayesflow.simulators.simulator import Simulator
from bayesflow.types import Tensor


class OnlineDataset(keras.utils.PyDataset):
    """
    A dataset that is generated on-the-fly.
    """

    def __init__(self, simulator: Simulator, batch_size: int, adapter: DataAdapter[any, PyTree[Tensor]], **kwargs):
        super().__init__(**kwargs)

        if keras.backend.backend() == "torch" and kwargs.get("use_multiprocessing"):
            # keras workaround: https://github.com/keras-team/keras/issues/19346
            import multiprocessing as mp

            mp.set_start_method("spawn", force=True)

        self.batch_size = batch_size
        self.adapter = adapter
        self.simulator = simulator

    def __getitem__(self, item: int):
        batch = self.simulator.sample((self.batch_size,))

        if self.adapter is not None:
            batch = self.adapter.configure(batch)

        return batch

    @property
    def num_batches(self):
        # infinite dataset
        return None
