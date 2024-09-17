import keras
import numpy as np
import os
import pathlib as pl

from bayesflow.data_adapters import DataAdapter


class NumpyDataset(keras.utils.PyDataset):
    """
    A dataset used to load numpy files from disk.
    The training strategy will be offline.

    By default, the expected file structure is as follows:
    root
    ├── parameter_name_1
    │   ├── sample_1.npy
    │   ├── ...
    │   └── sample_n.npy
    ├── parameter_name_2
    │   ├── sample_1.npy
    │   ├── ...
    │   └── sample_n.npy
    └── ...

    where each numpy file contains a sample of a single parameter (i.e., a numpy array).

    """

    def __init__(
        self,
        root: os.PathLike,
        *,
        pattern: str = "*.npy",
        batch_size: int,
        data_adapter: DataAdapter | None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.root = pl.Path(root)
        self.data_adapter = data_adapter

        # TODO: the assumption on the file structure is a bit strong
        #  we should relax this assumption in the future or provide better customization
        #  via the pattern arguments
        parameter_names = list(map(str, self.root.glob("*/")))

        self.files = {
            parameter_name: list(map(str, self.root.glob(f"{parameter_name}/{pattern}")))
            for parameter_name in parameter_names
        }

        self.shuffle()

    def __getitem__(self, item):
        if not 0 <= item < self.num_batches:
            raise IndexError(f"Index {item} is out of bounds for dataset with {self.num_batches} batches.")

        batch = {}

        for parameter_name, files in self.files.items():
            files = files[item * self.batch_size : (item + 1) * self.batch_size]

            samples = []
            for file in files:
                samples.append(np.load(file))

            batch[parameter_name] = np.stack(samples)

        if self.data_adapter is not None:
            batch = self.data_adapter.configure(batch)

        return batch

    def on_epoch_end(self):
        self.shuffle()

    @property
    def num_batches(self):
        n = len(next(iter(self.files.values())))
        return int(np.ceil(n / self.batch_size))

    def shuffle(self):
        permutation = np.random.permutation(len(next(iter(self.files.values()))))
        self.files = {parameter_name: [files[i] for i in permutation] for parameter_name, files in self.files.items()}
