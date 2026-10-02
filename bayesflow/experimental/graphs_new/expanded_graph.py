from copy import deepcopy
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import networkx as nx

from .inverted_graph import InvertedGraph

if TYPE_CHECKING:
    from .simulation_graph import SimulationGraph

SimulationNode: TypeAlias = str


# expanded nodes have their own class, storing the original node name and an index
# storing which groups it belongs to
class ExpandedNode(NamedTuple):
    original: str
    index: tuple[tuple[str, int], ...] = ()


class ExpandedGraph(nx.DiGraph):
    def __init__(self, incoming_graph_data=None, *, simulation_graph: "SimulationGraph | None" = None, **kwargs):
        super().__init__(incoming_graph_data, **kwargs)
        self.simulation_graph = deepcopy(simulation_graph)

    def invert(self, ordering: list[SimulationNode]):
        inverted = nx.DiGraph()
        inverted.add_nodes_from(node for node in self if self.out_degree(node) == 0)

        for variable in ordering:
            for node in instances(self, variable):
                parents = minimal_conditions(self, node, known=set(inverted))
                inverted.add_node(node)
                inverted.add_edges_from((parent, node) for parent in parents)

        return InvertedGraph(inverted, expanded_graph=self)


def minimal_conditions(graph: nx.DiGraph, node: ExpandedNode, known: set[ExpandedNode]) -> set[ExpandedNode]:
    kept = set(known)

    for candidate in sorted(known):
        without = kept - {candidate}
        dropped = known - without
        if nx.is_d_separator(graph, {node}, dropped, without):
            kept = without

    return kept


def instances(graph, variable: SimulationNode) -> list[ExpandedNode]:
    return [node for node in graph if node.original == variable]
