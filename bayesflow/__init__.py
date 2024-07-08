from . import (
    approximators,
    configurators,
    datasets,
    diagnostics,
    distributions,
    networks,
    simulators,
    utils,
)

from .approximators import Approximator
from .datasets import OfflineDataset, OnlineDataset

import keras

if keras.backend.backend() == "torch":
    # turn off gradients by default
    import torch

    torch.autograd.set_grad_enabled(False)

    # clean up namespace
    del keras
    del torch
