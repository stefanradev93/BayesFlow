import keras
import numpy as np
import pytest

from bayesflow.adapters import Adapter
from bayesflow.experimental.graphical_approximator import GraphicalApproximator
from bayesflow.experimental.graphical_approximator.encoder import ExchangeableEncoder
from bayesflow.experimental.graphical_simulator.samples import concatenate, to_tensors
from bayesflow.networks import ConsistencyModel, CouplingFlow, DiffusionModel, FlowMatching, StableConsistencyModel

from .conftest import approximator_for


EXAMPLES = [
    ("single_level_simulator", "single_level_approximator"),
    ("two_level_simulator", "two_level_approximator"),
    ("three_level_simulator", "three_level_approximator"),
    ("two_data_nodes_simulator", "two_data_nodes_approximator"),
    ("crossed_design_irt_simulator", "crossed_design_irt_approximator"),
    ("survey_simulator", "survey_approximator"),
]


@pytest.mark.parametrize(("simulator", "approximator"), EXAMPLES)
def test_fit_sample(simulator, approximator, request):
    simulator = request.getfixturevalue(simulator)
    approximator = request.getfixturevalue(approximator)
    approximator.compile()

    metrics = approximator.compute_metrics(**approximator.to_batch(simulator.sample(2)))
    assert "loss" in metrics

    approximator.fit(simulator=simulator, batch_size=2, num_batches=1, epochs=1)
    approximator.fit(dataset=simulator.sample(4), batch_size=2, epochs=1)

    observed = simulator.sample(3)
    assert_samples_match(approximator, observed, approximator.sample(num_samples=5, conditions=observed), 5)

    log_prob = approximator.log_prob(observed)
    assert log_prob.shape == (3,)
    assert np.all(np.isfinite(log_prob))


def assert_samples_match(approximator, observed, samples, num_samples):
    latent_nodes = set()
    for target_nodes in approximator.networks:
        for node in target_nodes:
            latent_nodes.add(node)

            for column, ids in observed[node].indices.items():
                np.testing.assert_array_equal(samples[node].indices[column], ids)

            assert set(samples[node].values) == set(observed[node].values)

            for variable, values in observed[node].values.items():
                assert samples[node].values[variable].shape == (len(values), num_samples, values.shape[-1])
                assert isinstance(samples[node].values[variable], np.ndarray)

    assert set(samples) == latent_nodes


NETWORKS = {
    "flow_matching": FlowMatching,
    "diffusion": DiffusionModel,
    "consistency": lambda: ConsistencyModel(total_steps=10),
    "stable_consistency": StableConsistencyModel,
}


@pytest.mark.parametrize("simulator", ["two_level_simulator", "crossed_design_irt_simulator"])
@pytest.mark.parametrize("network", NETWORKS)
def test_fit_sample_networks(simulator, network, request):
    simulator = request.getfixturevalue(simulator)
    approximator = approximator_for(simulator, NETWORKS[network])
    approximator.compile()

    approximator.fit(simulator=simulator, batch_size=2, num_batches=1, epochs=1)

    observed = simulator.sample(3)
    assert_samples_match(approximator, observed, approximator.sample(num_samples=5, conditions=observed), 5)


def test_sample_tensor_format(two_level_simulator, two_level_approximator):
    two_level_approximator.build_from_data(two_level_approximator.to_batch(two_level_simulator.sample(2)))

    samples = two_level_approximator.sample(num_samples=5, conditions=two_level_simulator.sample(3), format="tensor")

    assert set(samples) == {"hyper_mean", "hyper_std", "local_mean", "shared_std"}
    assert samples["hyper_mean"].shape == (3, 5, 1)
    assert samples["local_mean"].shape == (3, 5, 6, 1)


def test_sample_tensor_format_raises_for_ragged_datasets(crossed_design_irt_simulator, crossed_design_irt_approximator):
    approximator = crossed_design_irt_approximator
    approximator.build_from_data(approximator.to_batch(crossed_design_irt_simulator.sample(2)))
    observed = crossed_design_irt_simulator.sample(3)

    with pytest.raises(ValueError, match="node 'students'"):
        approximator.sample(num_samples=5, conditions=observed, format="tensor")

    posterior = approximator.sample(num_samples=5, conditions=observed)
    num_students = np.sum(observed["students"].indices["dataset"] == 1)

    assert to_tensors(posterior, datasets=1)["student_ability"].shape == (1, 5, num_students, 1)


def test_save_and_load_flow_matching(tmp_path, two_level_simulator):
    approximator = approximator_for(two_level_simulator, FlowMatching)
    approximator.build_from_data(approximator.to_batch(two_level_simulator.sample(2)))

    approximator.save(tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert len(loaded.weights) == len(approximator.weights)
    for loaded_weight, weight in zip(loaded.weights, approximator.weights):
        np.testing.assert_allclose(keras.ops.convert_to_numpy(loaded_weight), keras.ops.convert_to_numpy(weight))


@pytest.mark.parametrize("network", ["consistency", "stable_consistency"])
def test_log_prob_raises_for_consistency_models(network, two_level_simulator):
    approximator = approximator_for(two_level_simulator, NETWORKS[network])
    approximator.build_from_data(approximator.to_batch(two_level_simulator.sample(2)))

    with pytest.raises(NotImplementedError, match="no tractable density"):
        approximator.log_prob(two_level_simulator.sample(2))


@pytest.mark.parametrize(("simulator", "approximator"), EXAMPLES)
def test_padding_preserves_loss(simulator, approximator, request):
    simulator = request.getfixturevalue(simulator)
    approximator = request.getfixturevalue(approximator)

    simulation = simulator.sample(5)
    batch = approximator.to_batch(simulation)
    approximator.build_from_data(batch)

    # fit converts the batches to tensors, so the padding weights are on the same device as the loss
    padded_batch = keras.tree.map_structure(keras.ops.convert_to_tensor, approximator.to_batch(simulation, padded=True))

    plain = approximator.compute_metrics(**batch, stage="validation")
    padded = approximator.compute_metrics(**padded_batch, stage="validation")

    for name, value in plain.items():
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(padded[name]), keras.ops.convert_to_numpy(value), rtol=1e-4, atol=1e-4
        )


def test_fit_with_validation_data(two_level_simulator, two_level_approximator):
    two_level_approximator.compile()

    history = two_level_approximator.fit(
        dataset=two_level_simulator.sample(4), validation_data=two_level_simulator.sample(3), batch_size=2, epochs=1
    )

    assert "val_loss" in history.history


def test_log_prob_sums_rows_per_dataset(two_level_simulator, two_level_approximator):
    simulation = two_level_simulator.sample(5)
    two_level_approximator.build_from_data(two_level_approximator.to_batch(simulation))

    log_prob = two_level_approximator.log_prob(simulation)

    np.testing.assert_allclose(
        two_level_approximator.log_prob(concatenate([simulation] * 2)), np.tile(log_prob, 2), atol=1e-4
    )


def test_log_prob_with_adapter(two_level_simulator):
    adapter = Adapter().standardize(include="hyper_mean", mean=0.0, std=2.0)
    approximator = GraphicalApproximator(
        two_level_simulator.graph.expand().invert(),
        adapter=adapter,
        inference_networks=[CouplingFlow(), CouplingFlow()],
    )

    simulation = two_level_simulator.sample(2)
    approximator.build_from_data(approximator.to_batch(simulation))

    log_prob = approximator.log_prob(simulation)
    assert log_prob.shape == (2,)
    assert np.all(np.isfinite(log_prob))


@pytest.mark.parametrize(("simulator", "approximator"), EXAMPLES)
def test_serialization(simulator, approximator, request):
    approximator = request.getfixturevalue(approximator)

    restored = GraphicalApproximator.from_config(approximator.get_config())

    assert restored.networks == approximator.networks
    assert restored.condition_nodes == approximator.condition_nodes


def test_save_and_load(tmp_path, two_data_nodes_simulator, two_data_nodes_approximator):
    approximator = two_data_nodes_approximator
    approximator.build_from_data(approximator.to_batch(two_data_nodes_simulator.sample(2)))

    approximator.save(tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert loaded.networks == approximator.networks
    assert loaded.condition_nodes == approximator.condition_nodes
    assert loaded.variable_dims == approximator.variable_dims

    simulation = two_data_nodes_simulator.sample(2)
    np.testing.assert_allclose(loaded.log_prob(simulation), approximator.log_prob(simulation), rtol=1e-5)


def test_defaults(two_data_nodes_approximator):
    approximator = two_data_nodes_approximator

    assert isinstance(approximator.adapter, Adapter)
    assert approximator.data_nodes == ["w", "y"]

    # one encoder per network and data node it conditions on
    for network_encoders in approximator.encoders:
        assert sorted(network_encoders) == ["w", "y"]
        for encoder in network_encoders.values():
            assert isinstance(encoder, ExchangeableEncoder)


def test_non_amortizable_networks_are_sequential(crossed_design_irt_approximator):
    assert crossed_design_irt_approximator.sequential == ["questions"]
