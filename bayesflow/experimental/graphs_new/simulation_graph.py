from typing import Callable, TypeAlias
from .expanded_graph import ExpandedGraph, ExpandedNode
import networkx as nx

SimulationNode: TypeAlias = str


class SimulationGraph(nx.DiGraph):
    def __init__(self, meta_fn: Callable = None, **kwargs):
        super().__init__(**kwargs)

    def expand(self) -> ExpandedGraph:
        expanded = nx.relabel_nodes(nx.DiGraph(self), ExpandedNode)

        for variable in nx.lexicographical_topological_sort(self):
            if is_interior(self, variable) and is_exchangeable(self, variable):
                instances = [node for node in expanded if node.original == variable]

                for node in instances:
                    split(expanded, node)

        return ExpandedGraph(expanded, simulation_graph=self)


def split(graph: nx.DiGraph, node: ExpandedNode):
    subtree = {node} | nx.descendants(graph, node)
    inner_edges = list(graph.subgraph(subtree).edges)
    incoming_edges = [(u, w) for u, w in graph.in_edges(subtree) if u not in subtree]

    graph.remove_nodes_from(subtree)

    for k in (1, 2):

        def copy(w: ExpandedNode) -> ExpandedNode:
            return ExpandedNode(w.original, (*w.index, (node.original, k)))

        graph.add_nodes_from(copy(w) for w in subtree)
        graph.add_edges_from((copy(u), copy(w)) for u, w in inner_edges)
        graph.add_edges_from((u, copy(w)) for u, w in incoming_edges)

    return graph


def is_interior(graph: SimulationGraph, node: SimulationNode):
    return graph.in_degree(node) > 0 and graph.out_degree(node) > 0


def is_exchangeable(graph: SimulationGraph, node: SimulationNode):
    reps = graph.nodes[node].get("reps", 1)

    return isinstance(reps, str) or reps > 1
