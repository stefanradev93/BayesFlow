from . import (
    approximators,
    benchmarks,
    configurators,
    datasets,
    diagnostics,
    distributions,
    networks,
    simulators,
    utils,
)

from .approximators import ContinuousApproximator
from .datasets import OfflineDataset, OnlineDataset


def setup():
    # perform any necessary setup without polluting the namespace
    import keras
    import logging

    logging.getLogger().setLevel(logging.INFO)

    if keras.backend.backend() == "torch":
        # turn off gradients by default
        import torch

        torch.autograd.set_grad_enabled(False)


# call and clean up namespace
setup()
del setup
