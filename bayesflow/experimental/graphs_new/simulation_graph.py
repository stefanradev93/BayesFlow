from typing import Callable, TypeAlias

import networkx as nx

from .expanded_graph import ExpandedGraph, ExpandedNode, instances

SimulationNode: TypeAlias = str


class SimulationGraph(nx.DiGraph):
    def __init__(self, meta_fn: Callable | None = None, **kwargs):
        super().__init__(**kwargs)
        self.meta_fn = meta_fn

    def expand(self) -> ExpandedGraph:
        expanded = nx.relabel_nodes(nx.DiGraph(self), ExpandedNode)

        for variable in nx.lexicographical_topological_sort(self):
            if is_interior(self, variable) and is_exchangeable(self, variable):
                for node in instances(expanded, variable):
                    split(expanded, node)

        return ExpandedGraph(expanded, simulation_graph=self)


def split(graph: nx.DiGraph, node: ExpandedNode):
    # replaces the node and all its descendants by two copies, which share the incoming edges
    subtree = {node} | nx.descendants(graph, node)
    inner_edges = list(graph.subgraph(subtree).edges)

    incoming_edges = []
    for parent, child in graph.in_edges(subtree):
        if parent not in subtree:
            incoming_edges.append((parent, child))

    graph.remove_nodes_from(subtree)

    for k in (1, 2):
        copies = {}
        for member in subtree:
            copies[member] = ExpandedNode(member.original, (*member.index, (node.original, k)))

        graph.add_nodes_from(copies.values())

        for parent, child in inner_edges:
            graph.add_edge(copies[parent], copies[child])

        for parent, child in incoming_edges:
            graph.add_edge(parent, copies[child])


def is_interior(graph: SimulationGraph, node: SimulationNode):
    return graph.in_degree(node) > 0 and graph.out_degree(node) > 0


def is_exchangeable(graph: SimulationGraph, node: SimulationNode):
    reps = graph.nodes[node].get("reps", 1)

    return callable(reps) or isinstance(reps, str) or reps > 1


def data_nodes(graph: SimulationGraph) -> list[SimulationNode]:
    nodes = []

    for node in nx.lexicographical_topological_sort(graph):
        if graph.out_degree(node) == 0:
            nodes.append(node)

    return nodes
