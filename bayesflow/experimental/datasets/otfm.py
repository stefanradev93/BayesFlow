
import keras
import numpy as np

from bayesflow.experimental.types import Tensor
from bayesflow.experimental import utils


class OptimalTransportFlowMatchingDatasetWrapper(keras.utils.PyDataset):
    def __init__(self, dataset: keras.utils.PyDataset, sigma: float = 1.0, atol: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.sigma = sigma
        self.atol = atol

    def __getitem__(self, item):
        x1, c = self.dataset[item]
        x0 = keras.random.normal(keras.ops.shape(x1))

        x0, x1 = match(x0, x1, sigma=self.sigma, atol=self.atol)

        t = keras.random.uniform(keras.ops.shape(x1)[0], 0, 1)
        t = utils.expand_right(t, keras.ops.ndim(x1) - 1)

        x = t * x1 + (1 - t) * x0

        x = keras.ops.concatenate([x, t, c], axis=-1)
        y = x1 - x0

        return x, y

    def __len__(self):
        return len(self.dataset)


def match(x: Tensor, y: Tensor, **kwargs) -> (Tensor, Tensor):
    log_plan = _sinkhorn(x, y, **kwargs)
    indices = np.random.multinomial(1, keras.ops.exp(log_plan))
    return x, y[indices]


def _sinkhorn(x: Tensor, y: Tensor, sigma: float = 1.0, atol: float = 1e-8) -> Tensor:
    # compute Euclidean distance
    cost = x[:, None] - y[None, :]
    cost = keras.ops.reshape(cost, cost.shape[:2] + (-1,))
    cost = keras.ops.norm(cost, axis=2)

    # scale sigma by mean cost
    sigma = 0.5 * sigma * keras.ops.mean(cost)

    # initialize as gaussian kernel
    log_plan = -0.5 * cost / sigma

    converged = False
    while not converged:
        # apply Sinkhorn-Knopp
        log_plan = keras.ops.log_softmax(log_plan, axis=0)
        log_plan = keras.ops.log_softmax(log_plan, axis=1)

        # check convergence
        marginal = keras.ops.logsumexp(log_plan, axis=0)
        converged = keras.ops.all(marginal < atol)

    return log_plan
