from warnings import warn

from bayesflow.experimental.graphical_estimator.example_estimators import crossed_design_irt_estimator


def crossed_design_irt_approximator():
    warn(
        "crossed_design_irt_approximator is deprecated. Use crossed_design_irt_estimator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return crossed_design_irt_estimator()
