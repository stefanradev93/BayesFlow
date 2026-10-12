import keras
import numpy as np
import pytest

from bayesflow.adapters import Adapter
from bayesflow.experimental.graphical_approximator_new.batches import add_sequence_summaries, groupings, to_batch
from bayesflow.experimental.graphical_approximator_new import GraphicalApproximator
from bayesflow.experimental.graphical_approximator_new.encoder import ExchangeableEncoder
from bayesflow.experimental.graphs_new.simulation_graph import data_nodes
from bayesflow.networks import CouplingFlow


def batch_of(simulator, num_datasets):
    batch = to_batch(simulator.sample(num_datasets), simulator.graph.expand().invert(), Adapter())

    for node in batch:
        for variable, values in batch[node]["values"].items():
            batch[node]["values"][variable] = keras.ops.convert_to_numpy(values)

    return batch


def encoders(graph, width=4):
    sinks = data_nodes(graph.expanded_graph.simulation_graph)

    result = []
    for target_nodes in graph.networks():
        network_encoders = {}
        for node in graph.conditions(target_nodes):
            if node in sinks:
                network_encoders[node] = ExchangeableEncoder(width=width)
        result.append(network_encoders)

    return result


def conditions(batch, graph, network_encoders):
    inference_networks = []
    for target_nodes in graph.networks():
        inference_networks.append(CouplingFlow())

    approximator = GraphicalApproximator(graph, inference_networks=inference_networks, encoders=network_encoders)

    result = []
    for i in range(len(inference_networks)):
        result.append(approximator.network_conditions(batch, i))

    return result


def permute_rows(batch, node, order):
    for variable, values in batch[node]["values"].items():
        batch[node]["values"][variable] = values[order]

    for column, ids in batch[node]["indices"].items():
        batch[node]["indices"][column] = ids[order]

    for group in batch[node].get("groups", {}).values():
        group["ids"] = group["ids"][order]


def test_groupings(two_level_simulator, three_level_simulator, crossed_design_irt_simulator):
    assert groupings(two_level_simulator.graph, "y") == [(), ("locals",)]
    assert groupings(three_level_simulator.graph, "scores") == [(), ("classrooms",), ("classrooms", "students")]

    result = groupings(crossed_design_irt_simulator.graph, "observations")
    assert sorted(result) == [(), ("questions",), ("students",)]


def test_group_ids(crossed_design_irt_simulator):
    batch = batch_of(crossed_design_irt_simulator, 2)
    groups = batch["observations"]["groups"]
    rows = batch["observations"]["indices"]

    assert sorted(groups) == ["dataset", "questions", "students"]
    np.testing.assert_array_equal(groups["dataset"]["ids"], rows["dataset"])
    np.testing.assert_array_equal(groups["dataset"]["counts"], np.bincount(rows["dataset"]))

    num_students = np.bincount(batch["students"]["indices"]["dataset"])
    answers_per_question = groups["questions"]["counts"][groups["questions"]["ids"]]
    np.testing.assert_array_equal(answers_per_question, num_students[rows["dataset"]])


@pytest.mark.parametrize(
    "simulator",
    [
        "single_level_simulator",
        "two_level_simulator",
        "three_level_simulator",
        "crossed_design_irt_simulator",
        "two_data_nodes_simulator",
    ],
)
def test_one_condition_row_per_inferred_row(simulator, request):
    simulator = request.getfixturevalue(simulator)
    batch = batch_of(simulator, 3)
    graph = simulator.graph.expand().invert()

    result = conditions(batch, graph, encoders(graph))

    for i, target_nodes in enumerate(graph.networks()):
        rows = len(batch[target_nodes[0]]["indices"][target_nodes[0]])
        assert keras.ops.shape(result[i])[0] == rows


def test_condition_widths(two_level_simulator, crossed_design_irt_simulator, two_data_nodes_simulator):
    batch = batch_of(two_level_simulator, 2)
    graph = two_level_simulator.graph.expand().invert()

    result = conditions(batch, graph, encoders(graph))

    # hypers and shared: encoded data (4) and their number (1)
    assert keras.ops.shape(result[0]) == (2, 5)
    # locals: encoded data with the attached hypers and shared (4 + 1), and hyper_mean and hyper_std (2)
    assert keras.ops.shape(result[1]) == (12, 7)

    batch = batch_of(crossed_design_irt_simulator, 2)
    graph = crossed_design_irt_simulator.graph.expand().invert()

    result = conditions(batch, graph, encoders(graph))
    # schools: encoded data (4) and their number (1)
    assert keras.ops.shape(result[graph.networks().index(["schools"])]) == (2, 5)

    batch = batch_of(two_data_nodes_simulator, 2)
    graph = two_data_nodes_simulator.graph.expand().invert()

    result = conditions(batch, graph, encoders(graph))
    # hypers and shared: encoded w and y, each with their number (2 * (4 + 1))
    assert keras.ops.shape(result[0]) == (2, 10)
    # locals: encoded w and y, each with their number, and hyper_mean and hyper_std (2)
    assert keras.ops.shape(result[1]) == (12, 12)


def test_conditions_are_invariant_to_the_order_of_observations(crossed_design_irt_simulator):
    batch = batch_of(crossed_design_irt_simulator, 2)
    graph = crossed_design_irt_simulator.graph.expand().invert()
    network_encoders = encoders(graph)
    before = conditions(batch, graph, network_encoders)

    order = np.random.default_rng(0).permutation(len(batch["observations"]["indices"]["observations"]))
    permute_rows(batch, "observations", order)
    after = conditions(batch, graph, network_encoders)

    for i in range(len(before)):
        # up to the order of float32 sums
        np.testing.assert_allclose(before[i], after[i], rtol=1e-4, atol=1e-4)


def test_conditions_depend_on_which_observations_share_a_group(two_level_simulator):
    batch = batch_of(two_level_simulator, 2)
    graph = two_level_simulator.graph.expand().invert()
    network_encoders = encoders(graph)
    before = conditions(batch, graph, network_encoders)

    # move the observations of dataset 0 between its groups
    rows = np.flatnonzero(batch["y"]["indices"]["dataset"] == 0)
    y = batch["y"]["values"]["y"]
    y[rows] = y[np.random.default_rng(0).permutation(rows)]
    after = conditions(batch, graph, network_encoders)

    # the summary for hypers depends on which observations share a group
    assert not np.allclose(before[0][0], after[0][0])
    # dataset 1 is untouched
    np.testing.assert_allclose(before[0][1], after[0][1], rtol=1e-4, atol=1e-4)


def test_conditions_are_paired_with_the_data(crossed_design_irt_simulator):
    batch = batch_of(crossed_design_irt_simulator, 2)
    graph = crossed_design_irt_simulator.graph.expand().invert()
    network_encoders = encoders(graph)
    students = graph.networks().index(["students"])
    before = conditions(batch, graph, network_encoders)

    # every student answers every question, so swapping the parameters of questions keeps each
    # student's average question, but changes which question each response belongs to
    rows = np.flatnonzero(batch["questions"]["indices"]["dataset"] == 0)
    permuted = np.random.default_rng(0).permutation(rows)
    for variable, values in batch["questions"]["values"].items():
        values[rows] = values[permuted]
    after = conditions(batch, graph, network_encoders)

    dataset = batch["students"]["indices"]["dataset"]
    assert not np.allclose(before[students][dataset == 0], after[students][dataset == 0])
    np.testing.assert_allclose(before[students][dataset == 1], after[students][dataset == 1], rtol=1e-4, atol=1e-4)


def test_steps(crossed_design_irt_simulator):
    batch = batch_of(crossed_design_irt_simulator, 2)
    steps = batch["questions"]["steps"]
    datasets = batch["questions"]["indices"]["dataset"]

    # the questions of each dataset form one group, numbered from 0
    np.testing.assert_array_equal(steps["ids"], datasets)
    np.testing.assert_array_equal(steps["counts"], np.bincount(datasets))

    for group in range(2):
        rows = np.flatnonzero(steps["ids"] == group)
        np.testing.assert_array_equal(steps["positions"][rows], np.arange(len(rows)))

    # every pair of rows within a group, the earlier one first
    assert len(steps["earlier"]) == np.sum(steps["positions"])
    np.testing.assert_array_equal(steps["ids"][steps["earlier"]], steps["ids"][steps["later"]])
    assert np.all(steps["positions"][steps["earlier"]] < steps["positions"][steps["later"]])


def test_sequence_summaries_only_see_earlier_rows(crossed_design_irt_simulator):
    batch = batch_of(crossed_design_irt_simulator, 2)
    positions = batch["questions"]["steps"]["positions"]
    ids = batch["questions"]["steps"]["ids"]
    conditions = keras.ops.zeros((len(positions), 3))
    before = keras.ops.convert_to_numpy(add_sequence_summaries(batch, "questions", conditions))

    # change the third question of the first dataset
    changed = np.flatnonzero((ids == 0) & (positions == 2))[0]
    for values in batch["questions"]["values"].values():
        values[changed] += 10.0
    after = keras.ops.convert_to_numpy(add_sequence_summaries(batch, "questions", conditions))

    # up to the order of float32 sums
    affected = np.flatnonzero(np.any(~np.isclose(before, after, atol=1e-4), axis=-1))
    np.testing.assert_array_equal(affected, np.flatnonzero((ids == 0) & (positions > 2)))


def test_encoder_serialization():
    encoder = ExchangeableEncoder(width=8, depth=3, activation="tanh")
    restored = ExchangeableEncoder.from_config(encoder.get_config())

    assert (restored.width, restored.depth, restored.activation) == (8, 3, "tanh")
