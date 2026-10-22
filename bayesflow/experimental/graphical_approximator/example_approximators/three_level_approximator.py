from warnings import warn

from bayesflow.experimental.graphical_estimator.example_estimators import three_level_estimator


def three_level_approximator():
    warn(
        "three_level_approximator is deprecated. Use three_level_estimator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return three_level_estimator()
