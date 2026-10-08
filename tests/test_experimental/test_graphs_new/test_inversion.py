import math

import numpy as np
import pytest

from bayesflow.experimental.graphs_new import InvertedGraph, SimulationGraph
from bayesflow.experimental.graphs_new.factorizations import enumerate_factorizations, mean_repetitions

SIMULATORS = ["single_level_simulator", "two_level_simulator", "three_level_simulator", "crossed_design_irt_simulator"]


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_invert_selects_a_factorization(request, simulator):
    expanded = request.getfixturevalue(simulator).graph.expand()
    inverted = expanded.invert()

    assert isinstance(inverted, InvertedGraph)
    assert set(inverted.nodes) == set(expanded.nodes)


def test_outer_nodes_first_is_amortizable(two_level_simulator):
    inverted = two_level_simulator.graph.expand().invert(["hypers", "shared", "locals"])

    assert inverted.is_amortizable("locals")


def test_inner_nodes_first_is_not_amortizable(two_level_simulator):
    inverted = two_level_simulator.graph.expand().invert(["locals", "hypers", "shared"])

    assert not inverted.is_amortizable("locals")


def test_locals_condition_on_their_own_data_only(two_level_simulator):
    inverted = two_level_simulator.graph.expand().invert(["hypers", "shared", "locals"])

    for node in inverted:
        if node.original == "locals":
            data = [parent for parent in inverted.predecessors(node) if parent.original == "y"]
            assert [parent.index for parent in data] == [node.index]


def test_crossed_design_infers_smaller_factor_sequentially(crossed_design_irt_simulator):
    # 15 questions vs. 100-200 students per school
    inverted = crossed_design_irt_simulator.graph.expand().invert()

    assert inverted.is_amortizable("students")
    assert not inverted.is_amortizable("questions")


def test_enumerate_factorizations_count(crossed_design_irt_simulator):
    graph = crossed_design_irt_simulator.graph
    num_latent = sum(1 for node in graph if graph.out_degree(node) > 0)

    assert len(enumerate_factorizations(graph.expand())) == math.factorial(num_latent)


def test_mean_repetitions():
    graph = SimulationGraph(meta_fn=lambda: {"num_groups": 4})
    graph.add_node("fixed", reps=3)
    graph.add_node("from_meta", reps="num_groups")
    graph.add_node("from_callable", reps=lambda size: np.full(size, 7))
    graph.add_node("default")
    graph.add_node("data", reps=lambda combinations: np.ones(len(combinations)))
    for node in ["fixed", "from_meta", "from_callable", "default"]:
        graph.add_edge(node, "data")

    # data nodes are never inferred, so their reps are skipped and may need the combinations
    assert mean_repetitions(graph) == {"fixed": 3, "from_meta": 4, "from_callable": 7, "default": 1}
