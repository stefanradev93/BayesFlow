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

    def networks(self) -> list[list[SimulationNode]]:
        simulation_graph = self.expanded_graph.simulation_graph

        networks = []

        for variable in self.ordering:
            is_root = simulation_graph.in_degree(variable) == 0
            follows_root = networks and simulation_graph.in_degree(networks[-1][-1]) == 0

            if is_root and follows_root:
                networks[-1].append(variable)
            else:
                networks.append([variable])

        return networks

    def conditions(self, network: list[SimulationNode]) -> list[SimulationNode]:
        parents = set()

        for variable in network:
            for node in instances(self, variable):
                for parent in self.predecessors(node):
                    parents.add(parent.original)

        simulation_graph = self.expanded_graph.simulation_graph

        result = []
        for variable in nx.lexicographical_topological_sort(simulation_graph):
            if variable in parents and variable not in network:
                result.append(variable)

        return result

    def is_amortizable(self, variable: SimulationNode) -> bool:
        nodes = instances(self, variable)
        return not any(self.has_edge(u, w) for u in nodes for w in nodes)


def instances(graph, variable: SimulationNode) -> list["ExpandedNode"]:
    return [node for node in graph if node.original == variable]
