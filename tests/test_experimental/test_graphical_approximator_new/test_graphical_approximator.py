import numpy as np
import pytest
from keras.src.callbacks.history import History

from bayesflow.experimental.graphical_approximator_new.batches import SimulationDataset, to_batch

from .conftest import approximator_for

EXAMPLES = [
    ("single_level_simulator", "single_level_approximator"),
    ("two_level_simulator", "two_level_approximator"),
    ("three_level_simulator", "three_level_approximator"),
    ("two_data_nodes_simulator", "two_data_nodes_approximator"),
]


def batch_of(approximator, simulation):
    return to_batch(simulation, approximator.adapter, approximator.groupings)


@pytest.mark.parametrize(("simulator", "approximator"), EXAMPLES)
def test_fit_sample(simulator, approximator, request):
    simulator = request.getfixturevalue(simulator)
    approximator = request.getfixturevalue(approximator)
    approximator.compile()

    metrics = approximator.compute_metrics(**batch_of(approximator, simulator.sample(2)))
    assert "loss" in metrics

    assert isinstance(approximator.fit(simulator=simulator, batch_size=2, num_batches=1, epochs=1), History)

    simulations = []
    for i in range(4):
        simulations.append(simulator.sample(1))

    dataset = SimulationDataset(
        simulations, batch_size=2, to_batch=lambda simulation: batch_of(approximator, simulation)
    )
    assert isinstance(approximator.fit(dataset=dataset, epochs=1), History)

    observed = simulator.sample(3)
    samples = approximator.sample(num_samples=5, conditions=observed)

    latent_variables = set()
    for network in approximator.networks:
        for node in network:
            for variable, values in observed[node].values.items():
                latent_variables.add(variable)
                assert samples[variable].shape == (len(values), 5, values.shape[-1])
                assert isinstance(samples[variable], np.ndarray)

    assert set(samples) == latent_variables


@pytest.mark.parametrize(("simulator", "approximator"), EXAMPLES)
def test_serialization(simulator, approximator, request):
    from bayesflow.experimental.graphical_approximator_new import GraphicalApproximator

    approximator = request.getfixturevalue(approximator)

    config = approximator.get_config()
    assert isinstance(GraphicalApproximator.from_config(config), GraphicalApproximator)


def test_save_and_load(tmp_path, two_data_nodes_simulator, two_data_nodes_approximator):
    import keras

    batch = batch_of(two_data_nodes_approximator, two_data_nodes_simulator.sample(2))
    approximator = two_data_nodes_approximator
    approximator.build_from_data(batch)

    approximator.save(tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert loaded.networks == approximator.networks
    assert loaded.conditions == approximator.conditions
    assert loaded.variable_dims == approximator.variable_dims

    expected = approximator.compute_metrics(**batch, stage="validation")["loss"]
    actual = loaded.compute_metrics(**batch, stage="validation")["loss"]
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_defaults(two_data_nodes_approximator):
    from bayesflow.adapters import Adapter
    from bayesflow.experimental.graphical_approximator_new.encoder import ExchangeableEncoder

    approximator = two_data_nodes_approximator

    assert isinstance(approximator.adapter, Adapter)
    assert approximator.data_nodes == ["w", "y"]

    # one encoder per network and data node it conditions on
    for network_encoders in approximator.encoders:
        assert sorted(network_encoders) == ["w", "y"]
        for encoder in network_encoders.values():
            assert isinstance(encoder, ExchangeableEncoder)


def test_rejects_non_amortizable_networks(crossed_design_irt_simulator):
    with pytest.raises(NotImplementedError, match="not amortizable"):
        approximator_for(crossed_design_irt_simulator)


def test_batch_size_from_data(two_level_simulator, two_level_approximator):
    batch = batch_of(two_level_approximator, two_level_simulator.sample(3))

    assert two_level_approximator._batch_size_from_data(batch) == 3


@pytest.mark.tensorflow
def test_fit_with_dataset_tensorflow(single_level_simulator, single_level_approximator):
    single_level_approximator.compile()

    simulations = []
    for i in range(8):
        simulations.append(single_level_simulator.sample(1))

    dataset = SimulationDataset(
        simulations, batch_size=4, to_batch=lambda simulation: batch_of(single_level_approximator, simulation)
    )
    fit = single_level_approximator.fit(dataset=dataset, epochs=1)
    assert isinstance(fit, History)


@pytest.mark.tensorflow
def test_fit_with_simulator_tensorflow(single_level_simulator, single_level_approximator):
    single_level_approximator.compile()

    fit = single_level_approximator.fit(simulator=single_level_simulator, batch_size=4, num_batches=2, epochs=1)
    assert isinstance(fit, History)
