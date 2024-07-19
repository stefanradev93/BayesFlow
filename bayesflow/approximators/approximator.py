import keras
import multiprocessing as mp

from bayesflow.configurators import Configurator
from bayesflow.datasets import OnlineDataset
from bayesflow.simulators import Simulator
from bayesflow.utils import find_maximum_batch_size, logging

from .backend_approximators import BackendWorkflow


class Approximator(BackendWorkflow):
    def fit(
        self,
        batch_size: int = "auto",
        configurator: Configurator = None,
        dataset: keras.utils.PyDataset = None,
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

            logging.info("Fitting on provided dataset.")

            return super().fit(x=dataset, y=None, **kwargs)

        # user did not pass a dataset, so we need to build one
        if simulator is None:
            raise ValueError("Received no data to fit on. Please provide a dataset or a simulator.")

        logging.info("Building online dataset from simulator.")

        if batch_size == "auto":

            def gen_fn(bs):
                return simulator.sample((bs,))

            # use a conservative estimate since this does not factor in autograd memory usage
            batch_size = find_maximum_batch_size(gen_fn, start=2**2, stop=2**16)
            batch_size //= 4

            logging.info("Using batch_size={bs}", bs=batch_size)

        if workers == "auto":
            workers = mp.cpu_count()
            logging.info("Using workers={w}", w=workers)
        elif workers is None:
            workers = 1
            logging.info("Using a single worker.")

        dataset = OnlineDataset(
            simulator=simulator,
            batch_size=batch_size,
            configurator=configurator,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )

        return super().fit(x=dataset, y=None, **kwargs)
