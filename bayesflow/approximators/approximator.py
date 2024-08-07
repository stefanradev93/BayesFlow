import keras
import multiprocessing as mp

from bayesflow.data_adapters import DataAdapter
from bayesflow.datasets import OnlineDataset
from bayesflow.simulators import Simulator
from bayesflow.utils import find_batch_size, filter_kwargs, logging
from bayesflow.types import Shape

from .backend_approximators import BackendApproximator


class Approximator(BackendApproximator):
    def build(self, data_shapes: dict[str, Shape]) -> None:
        raise NotImplementedError

    @classmethod
    def build_data_adapter(cls, **kwargs) -> DataAdapter:
        # implemented by each respective architecture
        raise NotImplementedError

    @classmethod
    def build_dataset(
        cls,
        *,
        batch_size: int = "auto",
        batches_per_epoch: int,
        data_adapter: DataAdapter = "auto",
        memory_budget: str | int = "auto",
        simulator: Simulator,
        workers: int = "auto",
        use_multiprocessing: bool = False,
        max_queue_size: int = 32,
        **kwargs,
    ) -> OnlineDataset:
        if batch_size == "auto":
            batch_size = find_batch_size(memory_budget=memory_budget, sample=simulator.sample((1,)))
            logging.info(f"Using a batch size of {batch_size}.")

        if data_adapter == "auto":
            data_adapter = cls.build_data_adapter(**filter_kwargs(kwargs, cls.build_data_adapter))

        if workers == "auto":
            workers = mp.cpu_count()
            logging.info(f"Using {workers} data loading workers to fully leverage your CPU.")

        workers = workers or 1

        return OnlineDataset(
            simulator=simulator,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            data_adapter=data_adapter,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=max_queue_size,
        )

    def fit(self, *, dataset: keras.utils.PyDataset = None, simulator: Simulator = None, **kwargs):
        if dataset is None:
            if simulator is None:
                raise ValueError("Received no data to fit on. Please provide either a dataset or a simulator.")

            logging.info(f"Building dataset from simulator instance of {simulator.__class__.__name__}.")
            dataset = self.build_dataset(simulator=simulator, **filter_kwargs(kwargs, self.build_dataset))
        else:
            if simulator is not None:
                raise ValueError(
                    "Received conflicting arguments. Please provide either a dataset or a simulator, but not both."
                )

            logging.info(f"Fitting on dataset instance of {dataset.__class__.__name__}.")

        if not self.built:
            logging.info("Building on a test batch.")
            test_batch = dataset[0]
            self.build({key: keras.ops.shape(value) for key, value in test_batch.items()})

        return super().fit(dataset=dataset, **kwargs)
