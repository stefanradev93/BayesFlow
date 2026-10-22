from warnings import warn

from bayesflow.experimental.graphical_estimator.example_estimators import two_level_estimator


def two_level_approximator():
    warn(
        "two_level_approximator is deprecated. Use two_level_estimator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return two_level_estimator()
