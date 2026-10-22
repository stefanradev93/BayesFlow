import keras
import pytest

from bayesflow.experimental.graphical_approximator import GraphicalApproximator
from bayesflow.experimental.graphical_estimator import GraphicalEstimator
from bayesflow.experimental.graphical_simulator.example_simulators import single_level_simulator
from bayesflow.experimental.self_consistency import SemiSupervisedApproximator, SemiSupervisedEstimator
from bayesflow.networks import CouplingFlow, DeepSet


def test_graphical_approximator_alias():
    assert GraphicalApproximator.__bases__ == (GraphicalEstimator,)
    assert keras.saving.get_registered_name(GraphicalApproximator) == "bayesflow.experimental>GraphicalApproximator"

    with pytest.warns(DeprecationWarning, match="GraphicalApproximator.*GraphicalEstimator"):
        approximator = GraphicalApproximator(
            single_level_simulator().graph.invert(),
            inference_networks=[CouplingFlow()],
            summary_networks=[DeepSet(summary_dim=10)],
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


def test_graphical_approximator_deep_module_exports():
    from bayesflow.experimental.graphical_approximator.example_approximators import single_level_approximator
    from bayesflow.experimental.graphical_approximator.inference_conditions import (
        inference_conditions_by_network as old_conditions,
    )
    from bayesflow.experimental.graphical_approximator.inference_variables import (
        inference_variables_by_network as old_variables,
    )
    from bayesflow.experimental.graphical_approximator.non_exchangeable_wrapper import (
        NonExchangeableWrapper as OldWrapper,
    )
    from bayesflow.experimental.graphical_approximator.tensor_concatenation import concatenate as old_concatenate
    from bayesflow.experimental.graphical_estimator.inference_conditions import (
        inference_conditions_by_network as new_conditions,
    )
    from bayesflow.experimental.graphical_estimator.inference_variables import (
        inference_variables_by_network as new_variables,
    )
    from bayesflow.experimental.graphical_estimator.non_exchangeable_wrapper import (
        NonExchangeableWrapper as NewWrapper,
    )
    from bayesflow.experimental.graphical_estimator.tensor_concatenation import concatenate as new_concatenate

    assert old_conditions is new_conditions
    assert old_variables is new_variables
    assert OldWrapper is NewWrapper
    assert old_concatenate is new_concatenate

    with pytest.warns(DeprecationWarning, match="single_level_approximator.*single_level_estimator"):
        estimator = single_level_approximator()
    assert isinstance(estimator, GraphicalEstimator)
