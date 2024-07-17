import multiprocessing as mp

from bayesflow.configurators import BaseConfigurator as Configurator

from .backend_workflows import BackendWorkflow


class Workflow(BackendWorkflow):
    def fit(
        self,
        batch_size: int = "auto",
        configurator: Configurator = "auto",
        epochs: int = 8,
        use_multiprocessing: bool = True,
        workers: int = "auto",
        **kwargs,
    ):
        if batch_size == "auto":
            # TODO: set batch size such that cpu:0 or gpu:0 memory is fully used (depending on which device is used)
            #  or at most the full dataset size
            batch_size = min(len(...), 256)

        if workers == "auto":
            workers = mp.cpu_count()
        elif workers is None:
            workers = 1

        # TODO: make configurated dataset
        dataset = ...

        return super().fit(x=dataset, y=None, **kwargs)
