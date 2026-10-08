import numpy as np
import pytest


def approximator_for(simulator):
    from bayesflow.experimental.graphical_approximator_new import GraphicalApproximator
    from bayesflow.networks import CouplingFlow

    graph = simulator.graph.expand().invert()

    inference_networks = []
    for network in graph.networks():
        inference_networks.append(CouplingFlow())

    return GraphicalApproximator(graph, inference_networks=inference_networks)


# a second data node below locals, with a top-level function as reps so that it can be saved
def w_reps(size):
    return np.random.randint(3, 6, size=size)


def sample_w(local_mean):
    return {"w": np.random.normal(local_mean, 0.5, size=2)}


@pytest.fixture()
def single_level_simulator():
    from bayesflow.experimental.graphical_simulator_new.example_simulators import single_level_simulator

    return single_level_simulator()


@pytest.fixture()
def single_level_approximator(single_level_simulator):
    return approximator_for(single_level_simulator)


@pytest.fixture()
def two_level_simulator():
    from bayesflow.experimental.graphical_simulator_new.example_simulators import two_level_simulator

    return two_level_simulator()


@pytest.fixture()
def two_level_approximator(two_level_simulator):
    return approximator_for(two_level_simulator)


@pytest.fixture()
def two_level_repeated_roots_simulator():
    from bayesflow.experimental.graphical_simulator_new.example_simulators import two_level_simulator

    return two_level_simulator(repeated_roots=True)


@pytest.fixture()
def two_data_nodes_simulator():
    from bayesflow.experimental.graphical_simulator_new.example_simulators import two_level_simulator

    simulator = two_level_simulator()
    simulator.add_node("w", sample_fn=sample_w, reps=w_reps)
    simulator.add_edge("locals", "w")

    return simulator


@pytest.fixture()
def two_data_nodes_approximator(two_data_nodes_simulator):
    return approximator_for(two_data_nodes_simulator)


@pytest.fixture()
def three_level_simulator():
    from bayesflow.experimental.graphical_simulator_new.example_simulators import three_level_simulator

    return three_level_simulator()


@pytest.fixture()
def three_level_approximator(three_level_simulator):
    return approximator_for(three_level_simulator)


@pytest.fixture()
def crossed_design_irt_simulator():
    from bayesflow.experimental.graphical_simulator_new.example_simulators import crossed_design_irt_simulator

    return crossed_design_irt_simulator()


@pytest.fixture()
def crossed_design_irt_approximator(crossed_design_irt_simulator):
    return approximator_for(crossed_design_irt_simulator)
