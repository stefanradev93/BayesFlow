from bayesflow.experimental.graphical_simulator.example_simulators import two_level_simulator
from bayesflow.adapters import Adapter
from bayesflow.networks import DeepSet, CouplingFlow
from bayesflow.experimental.graphical_estimator import GraphicalEstimator


def two_level_estimator():
    simulator = two_level_simulator()

    adapter = Adapter()

    summary_networks = [
        DeepSet(summary_dim=10),
        DeepSet(summary_dim=20),
        DeepSet(summary_dim=30),
    ]
    inference_networks = [CouplingFlow(), CouplingFlow()]

    inverted_graph = simulator.graph.invert()
    estimator = GraphicalEstimator(
        inverted_graph, adapter=adapter, inference_networks=inference_networks, summary_networks=summary_networks
    )

    return estimator
