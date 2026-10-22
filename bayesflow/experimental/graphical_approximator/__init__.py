"""Deprecated compatibility aliases for :mod:`bayesflow.experimental.graphical_estimator`."""

from bayesflow.experimental.graphical_estimator import ExchangeableEncoder

from .graphical_approximator import GraphicalApproximator

__all__ = ["GraphicalApproximator", "ExchangeableEncoder"]
