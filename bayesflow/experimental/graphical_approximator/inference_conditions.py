"""Compatibility exports for :mod:`bayesflow.experimental.graphical_estimator.inference_conditions`."""

from bayesflow.experimental.graphical_estimator.inference_conditions import (
    apply_sequential_chain,
    apply_summary,
    compute_chain_steps,
    expand_to_prefix,
    gather_node_output,
    inference_condition_shapes_by_network,
    inference_conditions_by_network,
    permute_to_prefix,
)

__all__ = [
    "apply_sequential_chain",
    "apply_summary",
    "compute_chain_steps",
    "expand_to_prefix",
    "gather_node_output",
    "inference_condition_shapes_by_network",
    "inference_conditions_by_network",
    "permute_to_prefix",
]
