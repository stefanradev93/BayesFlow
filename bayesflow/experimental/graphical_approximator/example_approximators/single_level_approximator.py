from warnings import warn

from bayesflow.experimental.graphical_estimator.example_estimators import single_level_estimator


def single_level_approximator():
    warn(
        "single_level_approximator is deprecated. Use single_level_estimator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return single_level_estimator()
