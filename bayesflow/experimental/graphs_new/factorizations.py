import itertools
import numpy as np

from .expanded_graph import ExpandedGraph
from .inverted_graph import InvertedGraph


def enumerate_factorizations(expanded_graph: ExpandedGraph) -> list[InvertedGraph]:
    simulation_graph = expanded_graph.simulation_graph
    latent = sorted(v for v in simulation_graph if simulation_graph.out_degree(v) > 0)

    factorizations = []
    for ordering in itertools.permutations(latent):
        factorizations.append(expanded_graph.invert(list(ordering)))

    return factorizations


def select_factorizations(factorizations: list[InvertedGraph]) -> InvertedGraph:
    simulation_graph = factorizations[0].expanded_graph.simulation_graph
    mean_reps = mean_repetitions(simulation_graph)

    costs = []
    for factorization in factorizations:
        sequential_instances = 0

        for variable in simulation_graph:
            if not factorization.is_amortizable(variable):
                sequential_instances += mean_reps[variable]

            costs.append(sequential_instances)

    return factorizations[np.argmin(costs)]


def mean_repetitions(simulation_graph, num_samples: int = 1000) -> dict[str, float]:
    meta_fn = simulation_graph.meta_fn
    meta_samples = [meta_fn() for _ in range(num_samples)] if meta_fn else []

    mean_meta = {}
    if meta_samples:
        for key in meta_samples[0]:
            mean_meta[key] = np.mean([sample[key] for sample in meta_samples])

    mean_reps = {}
    for variable in simulation_graph:
        reps = simulation_graph.nodes[variable].get("reps", 1)
        mean_reps[variable] = mean_meta[reps] if isinstance(reps, str) else reps

    return mean_reps
