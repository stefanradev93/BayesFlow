
import ctypes
import keras
import multiprocessing as mp

from copy import copy
from typing import Callable

from bayesflow.experimental.types import Tensor
from .ordered_channel import OrderedChannel


class DataLoader:
    def __init__(self, fetch_fn: Callable[[int], list[Tensor]], workers: int = 1, max_queue_size: int = 128):
        self.fetch_fn = fetch_fn
        self.index = mp.Value(ctypes.c_uint64, 0)

        self.channel = OrderedChannel(max_queue_size)
        self.workers = [mp.Process(target=self._work, daemon=True) for _ in range(workers)]

        for worker in self.workers:
            worker.start()

    def _work(self):
        while True:
            with self.index.get_lock():
                # reserve this index
                index = copy(self.index.value)
                self.index.value += 1

            data = self.fetch_fn(index)
            self.channel.put(index, data)

    def get_batch(self, batch_size):
        # TODO: epoch edge conditions
        batch = []
        for i in range(batch_size):
            batch.append(self.channel.get())
        batch = self._convert_batch(batch)
        return batch

    def _convert_batch(self, batch: any) -> list:
        # TODO: do not assume list of tensors
        return [keras.ops.stack(batch[i][j] for i in range(len(batch))) for j in range(len(batch[0]))]

    def __del__(self):
        for worker in self.workers:
            worker.terminate()
