from typing import TYPE_CHECKING, TypeAlias

import networkx as nx

if TYPE_CHECKING:
    from .expanded_graph import ExpandedGraph, ExpandedNode

SimulationNode: TypeAlias = str


class InvertedGraph(nx.DiGraph):
    def __init__(
        self,
        incoming_graph_data=None,
        *,
        expanded_graph: "ExpandedGraph | None" = None,
        ordering: list[SimulationNode] | None = None,
        **kwargs,
    ):
        super().__init__(incoming_graph_data, **kwargs)
        self.expanded_graph = expanded_graph
        self.ordering = ordering

    def is_amortizable(self, variable: SimulationNode) -> bool:
        nodes = instances(self, variable)
        return not any(self.has_edge(u, w) for u in nodes for w in nodes)


def instances(graph, variable: SimulationNode) -> list["ExpandedNode"]:
    return [node for node in graph if node.original == variable]
