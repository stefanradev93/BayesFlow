import keras
import pytest

from bayesflow.experimental.graphical_approximator import GraphicalApproximator
from bayesflow.experimental.graphical_estimator import GraphicalEstimator
from bayesflow.experimental.graphical_simulator.example_simulators import single_level_simulator
from bayesflow.experimental.self_consistency import SemiSupervisedApproximator, SemiSupervisedEstimator
from bayesflow.networks import CouplingFlow


def test_graphical_approximator_alias():
    assert GraphicalApproximator.__bases__ == (GraphicalEstimator,)
    assert keras.saving.get_registered_name(GraphicalApproximator) == "bayesflow.experimental>GraphicalApproximator"

    with pytest.warns(DeprecationWarning, match="GraphicalApproximator.*GraphicalEstimator"):
        approximator = GraphicalApproximator(
            single_level_simulator().graph.expand().invert(),
            inference_networks=[CouplingFlow()],
        )

    assert isinstance(approximator, GraphicalEstimator)


def test_semi_supervised_approximator_alias():
    assert SemiSupervisedApproximator.__bases__ == (SemiSupervisedEstimator,)
    assert (
        keras.saving.get_registered_name(SemiSupervisedApproximator)
        == "bayesflow.experimental>SemiSupervisedApproximator"
    )

    with pytest.warns(DeprecationWarning, match="SemiSupervisedApproximator.*SemiSupervisedEstimator"):
        approximator = SemiSupervisedApproximator(approximators={})

    assert isinstance(approximator, SemiSupervisedEstimator)
    assert approximator.approximators == approximator.estimators == {}
