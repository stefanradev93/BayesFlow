import keras

from bayesflow.types import Tensor

from ..dispatch import find_cost

from .sinkhorn_knopp import sinkhorn_knopp, sinkhorn_log


def optimal_transport(
    x1: Tensor, x2: Tensor, method: str = "sinkhorn_knopp", cost: str | Tensor = "euclidean", seed=None, **kwargs
) -> (Tensor, Tensor):
    """Matches elements from x2 onto x1, such that the transport cost between them is minimized,
    according to the method and cost matrix used.

    Depending on the method used, elements in either tensor may be permuted, dropped, or duplicated
    such that the assignment is optimal.

    :param x1: Tensor of shape (n, ...)
        Samples from the first distribution.

    :param x2: Tensor of shape (m, ...)
        Samples from the second distribution.

    :param method: Method used to compute the transport cost.
        Default: 'sinkhorn_knopp'

    :param cost: Method used to compute the transport cost.
        You may also pass the cost matrix of shape (n, m) directly.
        Default: 'euclidean'

    :param seed: Random seed used in randomized selection methods.
        Default: None

    :param kwargs: Additional keyword arguments passed to the optimization method.

    :return: Tensors of shapes (n, ...) and (m, ...)
        x1 and x2 in optimal transport permutation order.
    """
    match method.lower():
        case "hungarian":
            raise NotImplementedError("TODO")
        case "sinkhorn" | "sinkhorn_knopp":
            cost_matrix = find_cost(cost, x1, x2)
            transport_plan = sinkhorn_knopp(cost_matrix, **kwargs)
            indices = keras.random.categorical(transport_plan, num_samples=1, seed=seed)
            indices = keras.ops.squeeze(indices, axis=1)
            x1 = keras.ops.take(x1, indices, axis=0)
        case "sinkhorn_log":
            cost_matrix = find_cost(cost, x1, x2)
            transport_plan = sinkhorn_log(cost_matrix, **kwargs)
            indices = keras.random.categorical(transport_plan, num_samples=1, seed=seed)
            indices = keras.ops.squeeze(indices)
            x1 = keras.ops.take(x1, indices, axis=0)
        case "random":
            m = keras.ops.shape(x2)[0]
            indices = keras.random.randint((m,))
            x1 = keras.ops.take(x1, indices, axis=0)
        case other:
            raise ValueError(f"Unsupported method name: '{other}'.")

    return x1, x2
