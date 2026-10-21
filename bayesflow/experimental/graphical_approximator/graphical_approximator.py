from functools import wraps
from warnings import warn

from bayesflow.experimental.graphical_estimator import GraphicalEstimator
from bayesflow.utils.serialization import serializable


@serializable("bayesflow.experimental")
class GraphicalApproximator(GraphicalEstimator):
    """Deprecated alias for :class:`~bayesflow.experimental.graphical_estimator.GraphicalEstimator`."""

    @wraps(GraphicalEstimator.__init__)
    def __init__(self, *args, **kwargs):
        warn(
            "GraphicalApproximator is deprecated. Use "
            "bayesflow.experimental.graphical_estimator.GraphicalEstimator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
