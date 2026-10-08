import networkx as nx
import numpy as np
import pytest

from bayesflow.experimental.graphical_simulator_new import GraphicalSimulator
from bayesflow.experimental.graphical_simulator_new.samples import concatenate, split

SIMULATORS = ["single_level_simulator", "two_level_simulator", "three_level_simulator", "crossed_design_irt_simulator"]


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_one_row_per_sample(request, simulator):
    simulation = request.getfixturevalue(simulator).sample(5)

    for samples in simulation.values():
        for values in samples.values.values():
            assert values.ndim == 2
            assert len(values) == len(samples.indices)


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_index_columns(request, simulator):
    simulator = request.getfixturevalue(simulator)
    simulation = simulator.sample(5)

    for node in simulator.graph:
        indices = simulation[node].indices
        ancestors = nx.ancestors(simulator.graph, node)

        assert set(indices.columns) == {"dataset", node} | ancestors
        assert list(indices[node]) == list(range(len(indices)))
        assert set(indices["dataset"]) <= set(range(5))


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_every_dataset_has_every_root(request, simulator):
    simulator = request.getfixturevalue(simulator)
    simulation = simulator.sample(5)

    for node in simulator.graph:
        if simulator.graph.in_degree(node) == 0:
            assert list(simulation[node].indices["dataset"]) == list(range(5))


def test_fixed_reps(two_level_simulator):
    simulation = two_level_simulator.sample(5)

    assert len(simulation["hypers"].indices) == 5
    assert len(simulation["locals"].indices) == 5 * 6
    assert len(simulation["y"].indices) == 5 * 6 * 10


def test_meta_override(single_level_simulator):
    simulation = single_level_simulator.sample(5, meta={"N": 7})

    assert len(simulation["likelihood"].indices) == 5 * 7


def test_callable_reps():
    simulator = GraphicalSimulator()
    simulator.add_node("group", sample_fn=lambda: {"mu": np.random.normal()})
    simulator.add_node("obs", sample_fn=lambda mu: {"y": mu}, reps=lambda size: np.arange(size) + 1)
    simulator.add_edge("group", "obs")

    simulation = simulator.sample(4)

    assert list(simulation["obs"].indices.groupby("group").size()) == [1, 2, 3, 4]


def first_items(combinations, num_items):
    # the reps see every combination of group and item, and meta
    assert len(combinations) == combinations["group"].nunique() * num_items

    # the k-th group of a dataset observes its first k + 1 items
    return (combinations["item"] % num_items <= combinations["group"] % 3).astype(int).to_numpy()


def test_reps_receive_combinations_and_meta():
    simulator = GraphicalSimulator(meta_fn=lambda: {"num_items": 4})
    simulator.add_node("group", sample_fn=lambda: {"mu": np.random.normal()}, reps=3)
    simulator.add_node("item", sample_fn=lambda: {"b": np.random.normal()}, reps="num_items")
    simulator.add_node("obs", sample_fn=lambda mu, b: {"y": mu + b}, reps=first_items)
    simulator.add_edge("group", "obs")
    simulator.add_edge("item", "obs")

    simulation = simulator.sample(2)
    obs = simulation["obs"].indices

    # combinations with zero reps have no rows
    assert list(obs.groupby("group").size()) == [1, 2, 3, 1, 2, 3]
    np.testing.assert_array_equal(obs["obs"], np.arange(len(obs)))


def test_crossed_samples_join_within_school(crossed_design_irt_simulator):
    simulation = crossed_design_irt_simulator.sample(5)
    observations = simulation["observations"].indices

    # each observation pairs a question and a student of the same school
    question_school = simulation["questions"].indices["schools"].to_numpy()[observations["questions"]]
    student_school = simulation["students"].indices["schools"].to_numpy()[observations["students"]]

    assert np.array_equal(question_school, observations["schools"])
    assert np.array_equal(student_school, observations["schools"])


def sample_parent():
    return {"x": np.random.normal()}


def sample_parents(batch_size):
    return {"x": np.random.normal(size=batch_size)}


def copy_parent(x):
    return {"x_copy": x}


def copy_model(is_batched):
    """A model whose child copies its parent's value, so that the alignment of parents and children is visible."""
    simulator = GraphicalSimulator()

    if is_batched:
        simulator.add_node("parent", sample_fn=sample_parents, is_batched=True)
    else:
        simulator.add_node("parent", sample_fn=sample_parent)

    simulator.add_node("child", sample_fn=copy_parent, reps=3, is_batched=is_batched)
    simulator.add_edge("parent", "child")

    return simulator


@pytest.mark.parametrize("is_batched", [False, True])
def test_children_receive_their_parents_values(is_batched):
    simulation = copy_model(is_batched).sample(4)

    parent_row = simulation["child"].indices["parent"].to_numpy()
    expected = simulation["parent"].values["x"][parent_row]

    assert np.array_equal(simulation["child"].values["x_copy"], expected)


@pytest.mark.parametrize("is_batched", [False, True])
def test_vector_valued_variables(is_batched):
    simulator = GraphicalSimulator()

    if is_batched:
        simulator.add_node("prior", sample_fn=lambda batch_size: {"beta": np.zeros((batch_size, 3))}, is_batched=True)
    else:
        simulator.add_node("prior", sample_fn=lambda: {"beta": np.zeros(3)})

    assert simulator.sample(4)["prior"].values["beta"].shape == (4, 3)


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_split(request, simulator):
    simulation = request.getfixturevalue(simulator).sample(4)
    parts = split(simulation)

    assert len(parts) == 4

    # every part is a simulation of a single dataset
    for part in parts:
        for node, samples in part.items():
            assert (samples.indices["dataset"] == 0).all()
            np.testing.assert_array_equal(samples.indices[node], np.arange(len(samples.indices)))

    # concatenating the parts restores the simulation
    restored = concatenate(parts)
    for node, samples in simulation.items():
        np.testing.assert_array_equal(restored[node].indices.to_numpy(), samples.indices.to_numpy())
        for variable, values in samples.values.items():
            np.testing.assert_array_equal(restored[node].values[variable], values)
