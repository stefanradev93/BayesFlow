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


# a survey, where every square is surveyed in one window of years, so obs has no row for most (square, year) pairs
def squares_reps(size):
    return np.random.randint(1, 5, size=size)


def surveyed(combinations, num_years):
    squares, square = np.unique(combinations["squares"], return_inverse=True)
    year = combinations["years"] % num_years

    first = np.random.randint(0, num_years, size=len(squares))
    last = first + np.random.randint(0, num_years - first)

    return ((year >= first[square]) & (year <= last[square])).astype(int).to_numpy()


@pytest.fixture()
def survey_simulator():
    from bayesflow.experimental.graphical_simulator_new import GraphicalSimulator

    simulator = GraphicalSimulator(meta_fn=lambda: {"num_years": 5})
    simulator.add_node("globals", sample_fn=lambda: {"mu": np.random.normal()})
    simulator.add_node("squares", sample_fn=lambda mu: {"alpha": np.random.normal(mu)}, reps=squares_reps)
    simulator.add_node("years", sample_fn=lambda: {"beta": np.random.normal()}, reps="num_years")
    simulator.add_node("obs", sample_fn=lambda alpha, beta: {"count": np.random.normal(alpha + beta)}, reps=surveyed)
    simulator.add_edge("globals", "squares")
    simulator.add_edge("globals", "years")
    simulator.add_edge("squares", "obs")
    simulator.add_edge("years", "obs")

    return simulator


@pytest.fixture()
def survey_approximator(survey_simulator):
    return approximator_for(survey_simulator)


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
