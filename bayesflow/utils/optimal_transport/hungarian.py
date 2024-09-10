from bayesflow.types import Tensor


def hungarian(x1, x2, *aux, cost: str | Tensor = "euclidean", **kwargs) -> (Tensor, Tensor):
    raise NotImplementedError("Hungarian algorithm is not yet implemented.")
