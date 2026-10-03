import numpy as np
import pytest

from bayesflow.experimental.graphs_new import ExpandedGraph, SimulationGraph
from bayesflow.experimental.graphs_new.simulation_graph import is_exchangeable

SIMULATORS = ["single_level_simulator", "two_level_simulator", "three_level_simulator", "crossed_design_irt_simulator"]


@pytest.mark.parametrize(
    "simulator, num_nodes",
    [
        ("single_level_simulator", 2),
        ("two_level_simulator", 6),
        ("three_level_simulator", 12),
        ("crossed_design_irt_simulator", 9),
    ],
)
def test_expand_num_nodes(request, simulator, num_nodes):
    expanded = request.getfixturevalue(simulator).graph.expand()

    assert isinstance(expanded, ExpandedGraph)
    assert expanded.number_of_nodes() == num_nodes


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_expand_preserves_dependencies(request, simulator):
    graph = request.getfixturevalue(simulator).graph
    expanded = graph.expand()

    assert {(u.original, w.original) for u, w in expanded.edges} == set(graph.edges)


def test_expand_splits_into_two_instances(two_level_simulator):
    expanded = two_level_simulator.graph.expand()

    locals_ = [node for node in expanded if node.original == "locals"]
    assert sorted(node.index for node in locals_) == [(("locals", 1),), (("locals", 2),)]

    # each copy of y belongs to exactly one copy of locals
    for node in locals_:
        children = list(expanded.successors(node))
        assert [child.original for child in children] == ["y"]
        assert children[0].index == node.index


def test_expand_skips_single_repetition(two_level_simulator):
    two_level_simulator.graph.nodes["locals"]["reps"] = 1

    assert two_level_simulator.graph.expand().number_of_nodes() == 4


@pytest.mark.parametrize(
    "reps, exchangeable",
    [(1, False), (5, True), ("num_groups", True), (lambda size: np.full(size, 3), True)],
)
def test_is_exchangeable(reps, exchangeable):
    graph = SimulationGraph()
    graph.add_node("group", reps=reps)

    assert is_exchangeable(graph, "group") == exchangeable
