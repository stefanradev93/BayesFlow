import keras
import pytest

from bayesflow.experimental.graphical_estimator.non_exchangeable_wrapper import NonExchangeableWrapper
from bayesflow.networks import CouplingFlow, DeepSet


@pytest.fixture
def built_estimator(crossed_design_irt_simulator, crossed_design_irt_estimator):
    data = crossed_design_irt_simulator.sample(2)
    # Keras treats the SimulationOutput dict subclass as a tree leaf.
    adapted = crossed_design_irt_estimator.adapter(dict(data))
    data_shapes = crossed_design_irt_estimator._data_shapes(adapted)
    crossed_design_irt_estimator.build(data_shapes)
    crossed_design_irt_estimator.compile()
    crossed_design_irt_estimator.compute_metrics(**adapted)

    return crossed_design_irt_estimator, adapted


def test_serialization(crossed_design_irt_estimator):
    wrapper = crossed_design_irt_estimator.inference_networks[1]
    config = wrapper.get_config()
    restored = NonExchangeableWrapper.from_config(config)
    assert isinstance(restored, NonExchangeableWrapper)


def test_build():
    wrapper = NonExchangeableWrapper(CouplingFlow(), DeepSet(summary_dim=8))
    wrapper.build((2, 4, 2))
    assert wrapper.built

    wrapper.build((2, 4, 2))  # early return for already-built wrapper
    wrapper2 = NonExchangeableWrapper(CouplingFlow(), DeepSet(summary_dim=8))
    wrapper2.build((2, 4, 2), conditions_shape=(2, 4, 12))
    assert wrapper2.built


def test_call(built_estimator):
    from bayesflow.experimental.graphical_estimator.inference_conditions import inference_conditions_by_network
    from bayesflow.experimental.graphical_estimator.inference_variables import inference_variables_by_network

    estimator, adapted = built_estimator
    wrapper = estimator.inference_networks[1]
    variables = inference_variables_by_network(estimator, adapted)
    conditions = inference_conditions_by_network(estimator, adapted, estimator.summary_registry)
    xz, cond = variables[1], conditions[1]
    noise = keras.ops.zeros_like(xz)

    result = wrapper(xz, conditions=cond, inverse=False, density=False)
    assert result.shape == xz.shape

    z, _ = wrapper(xz, conditions=cond, inverse=False, density=True)
    assert z.shape == xz.shape

    result = wrapper(noise, conditions=cond, inverse=True, density=False)
    assert result.shape == xz.shape

    values, _ = wrapper(noise, conditions=cond, inverse=True, density=True)
    assert values.shape == xz.shape

    samples = wrapper.sample(tuple(xz.shape)[:-1], conditions=cond)
    assert samples.shape == xz.shape

    log_prob = wrapper.log_prob(xz, conditions=cond)
    assert log_prob is not None


def test_prepare_inference_conditions(built_estimator):
    from bayesflow.experimental.graphical_estimator.inference_conditions import inference_conditions_by_network
    from bayesflow.experimental.graphical_estimator.inference_variables import inference_variables_by_network

    estimator, adapted = built_estimator
    wrapper = estimator.inference_networks[1]
    variables = inference_variables_by_network(estimator, adapted)
    conditions = inference_conditions_by_network(estimator, adapted, estimator.summary_registry)
    xz = variables[1]

    result = wrapper._prepare_inference_conditions(xz, i=0, conditions=None)
    assert result is not None

    single_step_cond = conditions[1][..., :1, :]
    result = wrapper._prepare_inference_conditions(xz, i=0, conditions=single_step_cond)
    assert result is not None
