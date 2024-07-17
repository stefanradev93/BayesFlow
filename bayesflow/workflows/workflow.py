import keras
import multiprocessing as mp

from bayesflow.datasets import OnlineDataset
from bayesflow.simulators import Simulator
from bayesflow.utils import find_maximum_batch_size

from .backend_workflows import BackendWorkflow


class Workflow(BackendWorkflow):
    def fit(
        self,
        batch_size: int = "auto",
        collate_fn: callable = None,
        dataset: keras.utils.PyDataset = None,
        epochs: int = 8,
        simulator: Simulator = None,
        use_multiprocessing: bool = True,
        workers: int = "auto",
        **kwargs,
    ):
        if dataset is None:
            # user should pass a simulator, so we can build a dataset
            if simulator is None:
                raise ValueError("Received no data to fit on. Either provide a dataset or a simulator.")

            if batch_size == "auto":

                def gen_fn(bs):
                    return simulator.sample((bs,))

                # use a conservative estimate since this does not factor in autograd memory usage
                batch_size = find_maximum_batch_size(gen_fn, start=2**2, stop=2**16)
                batch_size //= 4

            if workers == "auto":
                workers = mp.cpu_count()
            elif workers is None:
                workers = 1

            dataset = OnlineDataset(
                simulator=simulator,
                collate_fn=collate_fn,
                batch_size=batch_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
            )
        else:
            # user passed a dataset
            if simulator is not None:
                raise ValueError(
                    "Received conflicting arguments. Please provide either a dataset or a simulator, " "but not both."
                )

        return super().fit(x=dataset, y=None, **kwargs)
