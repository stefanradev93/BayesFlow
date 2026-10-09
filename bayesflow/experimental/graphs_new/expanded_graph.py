from copy import deepcopy
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import networkx as nx

from .inverted_graph import InvertedGraph
from .factorizations import enumerate_factorizations, select_factorization

if TYPE_CHECKING:
    from .simulation_graph import SimulationGraph

SimulationNode: TypeAlias = str


class ExpandedNode(NamedTuple):
    original: str
    index: tuple[tuple[str, int], ...] = ()


class ExpandedGraph(nx.DiGraph):
    def __init__(self, incoming_graph_data=None, *, simulation_graph: "SimulationGraph | None" = None, **kwargs):
        super().__init__(incoming_graph_data, **kwargs)
        self.simulation_graph = deepcopy(simulation_graph)

    def invert(self, ordering: list[SimulationNode] | None = None):
        if ordering is None:
            return select_factorization(enumerate_factorizations(self))

        inverted = nx.DiGraph()
        for node in self:
            if self.out_degree(node) == 0:
                inverted.add_node(node)

        for variable in ordering:
            for node in instances(self, variable):
                conditions = minimal_conditions(self, node, known=set(inverted))

                inverted.add_node(node)
                for condition in conditions:
                    inverted.add_edge(condition, node)

        return InvertedGraph(inverted, expanded_graph=self, ordering=ordering)


def minimal_conditions(graph: nx.DiGraph, node: ExpandedNode, known: set[ExpandedNode]) -> set[ExpandedNode]:
    # drops known nodes one by one, as long as the node stays independent of all dropped nodes
    conditions = set(known)

    for candidate in sorted(known):
        fewer = conditions - {candidate}
        if nx.is_d_separator(graph, {node}, known - fewer, fewer):
            conditions = fewer

    return conditions


def instances(graph, variable: SimulationNode) -> list[ExpandedNode]:
    return [node for node in graph if node.original == variable]
