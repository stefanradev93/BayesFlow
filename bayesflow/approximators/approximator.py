import keras
import multiprocessing as mp
import numpy as np

from bayesflow.configurators import Configurator
from bayesflow.datasets import OnlineDataset
from bayesflow.simulators import Simulator
from bayesflow.utils import format_bytes, keras_kwargs, logging, parse_bytes, size_of

from .backend_approximators import BackendApproximator


class Approximator(BackendApproximator):
    # TODO: move into backend approximator
    def __init__(self, *, configurator: Configurator = None, **kwargs):
        super().__init__(**keras_kwargs(kwargs))
        self.configurator = configurator
        # TODO: save fit config as attribute and serialize / deserialize with it?

    def fit(
        self,
        *,
        batch_size: int = "auto",
        configurator: Configurator = None,
        dataset: keras.utils.PyDataset = None,
        memory_budget: str | int = "auto",
        simulator: Simulator = None,
        workers: int = "auto",
        use_multiprocessing: bool = True,
        **kwargs,
    ):
        if dataset is not None:
            if simulator is not None:
                raise ValueError(
                    "Received conflicting arguments. Please provide either a dataset or a simulator, but not both."
                )

            logging.info(f"Fitting on dataset instance of {dataset.__class__.__name__}.")

            return super().fit(dataset=dataset, **kwargs)

        # user did not pass a dataset, so we need to build one
        if simulator is None:
            raise ValueError("Received no data to fit on. Please provide a dataset or a simulator.")

        logging.info(f"Building dataset from simulator instance of {simulator.__class__.__name__}.")

        if batch_size == "auto":
            if memory_budget == "auto":
                # TODO: fetch memory budget of first accelerator device, or of cpu
                memory_budget = ...
                raise NotImplementedError(
                    "Automatic memory budget is not yet supported. " "Please pass an explicit value."
                )
            elif isinstance(memory_budget, str):
                memory_budget = parse_bytes(memory_budget)

            sample_memory = size_of(simulator.sample((1,)))

            logging.info(f"Estimating memory footprint of one sample at {format_bytes(sample_memory)}.")

            # conservative estimate
            batch_size = memory_budget / (4 * sample_memory)

            # limit estimate to sensible range
            batch_size = int(np.clip(batch_size, 4, 8192))

            logging.info(f"Using a batch size of {batch_size} to fully leverage your memory.")

        if workers == "auto":
            workers = mp.cpu_count()
            logging.info(f"Using {workers} data loading workers to fully leverage your CPU.")

        configurator = configurator or self.configurator
        workers = workers or 1

        dataset = OnlineDataset(
            simulator=simulator,
            batch_size=batch_size,
            configurator=configurator,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )

        return super().fit(dataset=dataset, **kwargs)
