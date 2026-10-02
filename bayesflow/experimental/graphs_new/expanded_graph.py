from copy import deepcopy
from typing import TYPE_CHECKING, NamedTuple

import networkx as nx

if TYPE_CHECKING:
    from .simulation_graph import SimulationGraph


# expanded nodes have their own class, storing the original node name and an index
# storing which groups it belongs to
class ExpandedNode(NamedTuple):
    original: str
    index: tuple[tuple[str, int], ...] = ()


class ExpandedGraph(nx.DiGraph):
    def __init__(self, incoming_graph_data=None, *, simulation_graph: "SimulationGraph | None" = None, **kwargs):
        super().__init__(incoming_graph_data, **kwargs)
        self.simulation_graph = deepcopy(simulation_graph)
