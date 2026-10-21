from typing import TYPE_CHECKING, TypeVar

import sympy as sp

from bayesflow.types import Tensor

from .tensor_concatenation import concatenate

if TYPE_CHECKING:
    from .graphical_estimator import GraphicalEstimator

K = TypeVar("K", str, int)


def resolve_shapes(
    shapes: dict[K, tuple[int | sp.Expr, ...]], meta_dict: dict | None
) -> dict[K, tuple[int | sp.Expr, ...]]:
    """Substitutes sympy symbols in a dict of shape tuples using ``meta_dict``."""
    meta_dict = meta_dict or {}

    def resolve(x):
        if isinstance(x, sp.Expr):
            x = sp.simplify(x.subs(meta_dict))
            if isinstance(x, sp.Integer):
                return int(x)
        return x

    return {k: tuple(resolve(d) for d in shape) for k, shape in shapes.items()}


def inference_variables_by_network(estimator: "GraphicalEstimator", adapted_data: dict) -> dict[int, Tensor]:
    """
    Collects and concatenates the inference variables for each network.

    For each network, gathers the simulated parameter tensors assigned to that
    network (via ``GraphicalEstimator.network_composition``), optionally
    standardizes them, and concatenates them into a single tensor along the
    last axis.

    Parameters
    ----------
    estimator : GraphicalEstimator
        The estimator, providing network composition, variable names, and
        standardization layers.
    adapted_data : dict[str, Tensor]
        Dictionary mapping variable names to tensors, as produced by the
        adapter.

    Returns
    -------
    dict[int, Tensor]
        Mapping from network index to the concatenated inference variable
        tensor for that network. Networks whose variables are not present in
        ``adapted_data`` are omitted.
    """
    result = {}

    for network_idx in range(len(estimator.inference_networks)):
        names = []
        for node in estimator.network_composition[network_idx]:
            for name in estimator.variable_names[node]:
                names.append(name)

        if all(name in adapted_data for name in names):
            result[network_idx] = _collect_inference_variables(estimator, adapted_data, network_idx)

    return result


def _collect_inference_variables(
    estimator: "GraphicalEstimator",
    adapted_data: dict,
    network_idx: int,
) -> Tensor:
    """
    Returns the concatenated inference variable tensor for ``network_idx``.

    Raises ``KeyError`` if any required variable is missing from
    ``adapted_data``.
    """
    tensors = []

    for node in estimator.network_composition[network_idx]:
        for name in estimator.variable_names[node]:
            var = adapted_data[name]

            if name in estimator.standardize:
                var = estimator.standardize_layers[name](var, stage="validation")

            tensors.append(var)

    return concatenate(tensors)


def inference_variable_shapes_by_network(
    estimator: "GraphicalEstimator",
    data_shapes: dict | None = None,
    meta_dict: dict | None = None,
) -> dict[int, tuple[int | sp.Expr, ...]]:
    """
    Returns the concrete shape of the concatenated inference variable tensor
    for each network, with sympy symbols resolved from the data shapes.

    Parameters
    ----------
    estimator : GraphicalEstimator
        The estimator, providing graph structure and output shapes.
    data_shapes : dict, optional
        Concrete data shapes to resolve sympy symbols. Defaults to the
        symbolic output shapes stored on the estimator.
    meta_dict : dict, optional
        Additional symbol-to-value mappings for resolution.

    Returns
    -------
    dict[int, tuple]
        Mapping from network index to resolved shape tuple.
    """
    if not data_shapes:
        data_shapes = estimator.output_shapes

    data_shapes = resolve_shapes(data_shapes, meta_dict)
    meta_dict = estimator._meta_dict_from_data_shapes(data_shapes)

    return resolve_shapes(estimator.graph.inference_variable_shapes(), meta_dict)
