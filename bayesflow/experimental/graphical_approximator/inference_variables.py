"""Compatibility exports for :mod:`bayesflow.experimental.graphical_estimator.inference_variables`."""

from bayesflow.experimental.graphical_estimator.inference_variables import (
    inference_variable_shapes_by_network,
    inference_variables_by_network,
    resolve_shapes,
)

__all__ = ["inference_variable_shapes_by_network", "inference_variables_by_network", "resolve_shapes"]
