from bayesflow.types import Tensor

from .hungarian import hungarian
from .random import random
from .sinkhorn import sinkhorn


def optimal_transport(
    x1: Tensor, x2: Tensor, *aux: Tensor, method: str = "sinkhorn_knopp", **kwargs
) -> (Tensor, Tensor):
    """Matches elements from x2 onto x1, such that the transport cost between them is minimized, according to the method
    and cost matrix used.

    Depending on the method used, elements in either tensor may be permuted, dropped, duplicated, or otherwise modified,
    such that the assignment is optimal.

    Note: this is just a dispatch function that calls the appropriate optimal transport method.
    See the documentation of the respective method for more details.

    :param x1: Tensor of shape (n, ...)
        Samples from the first distribution.

    :param x2: Tensor of shape (m, ...)
        Samples from the second distribution.

    :param aux: Tensors of shape (n, ...)
        Auxiliary tensors to be permuted along with x1.
        Note that x2 is never permuted for all currently available methods.

    :param method: Method used to compute the transport cost.
        Default: 'sinkhorn_knopp'

    :param kwargs: Additional keyword arguments passed to the optimization method.

    :return: Tensors of shapes (n, ...) and (m, ...)
        x1 and x2 in optimal transport permutation order.
    """
    methods = {
        "hungarian": hungarian,
        "sinkhorn": sinkhorn,
        "sinkhorn_knopp": sinkhorn,
        "random": random,
    }

    method = method.lower()

    if method not in methods:
        raise ValueError(f"Unsupported method name: '{method}'.")

    return methods[method](x1, x2, *aux, **kwargs)
