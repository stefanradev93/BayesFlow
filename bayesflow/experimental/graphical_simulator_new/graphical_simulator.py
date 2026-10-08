import inspect
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd

from ..graphs_new import SimulationGraph
from .samples import NodeSamples, Simulation


class GraphicalSimulator:
    def __init__(self, meta_fn: Callable[[], dict[str, Any]] | None = None):
        self.graph = SimulationGraph(meta_fn=meta_fn)

    def add_node(
        self,
        node: str,
        sample_fn: Callable[..., dict[str, Any]],
        reps: int | str | Callable = 1,
        is_batched: bool = False,
    ):
        self.graph.add_node(node, sample_fn=sample_fn, reps=reps, is_batched=is_batched)

    def add_edge(self, from_node: str, to_node: str):
        self.graph.add_edge(from_node, to_node)

    def sample(self, num_datasets: int, meta: dict[str, Any] | None = None) -> Simulation:
        meta_fn = self.graph.meta_fn
        meta = (meta_fn() if meta_fn else {}) | (meta or {})

        simulation = {}

        for node in nx.lexicographical_topological_sort(self.graph):
            parents = {parent: simulation[parent] for parent in self.graph.predecessors(node)}
            sample_fn = self.graph.nodes[node]["sample_fn"]
            reps = self.graph.nodes[node]["reps"]
            is_batched = self.graph.nodes[node]["is_batched"]

            indices = assign_indices(node, parents, reps, meta, num_datasets)
            values = draw_values(sample_fn, parents, indices, meta, is_batched)

            simulation[node] = NodeSamples(indices, values)

        return simulation


def assign_indices(node, parents, reps, meta, num_datasets) -> pd.DataFrame:
    combinations = pd.DataFrame({"dataset": np.arange(num_datasets)})
    for samples in parents.values():
        combinations = combinations.merge(samples.indices)

    num_samples = draw_num_samples(reps, meta, combinations=combinations)
    indices = combinations.apply(np.repeat, repeats=num_samples).reset_index(drop=True)
    indices[node] = np.arange(len(indices))

    return indices


def draw_values(sample_fn, parents, indices, meta, is_batched) -> dict[str, np.ndarray]:
    num_samples = len(indices)
    parent_values = {}

    for parent, samples in parents.items():
        parent_sample = indices[parent].to_numpy()

        for variable, values in samples.values.items():
            parent_values[variable] = values[parent_sample]

    if is_batched:
        outputs = call_sample_fn(sample_fn, parent_values | meta | {"batch_size": num_samples, "indices": indices})
    else:
        outputs = draw_one_by_one(sample_fn, parent_values, meta, num_samples)

    values = {}
    for variable, value in outputs.items():
        values[variable] = np.reshape(value, (num_samples, -1))

    return values


def draw_one_by_one(sample_fn, parent_values, meta, num_samples) -> dict[str, np.ndarray]:
    outputs_per_sample = []

    for i in range(num_samples):
        parent_values_per_sample = {}
        for variable, values in parent_values.items():
            parent_values_per_sample[variable] = np.squeeze(values[i])

        outputs_per_sample.append(call_sample_fn(sample_fn, parent_values_per_sample | meta))

    outputs = {}
    for variable in outputs_per_sample[0]:
        outputs[variable] = np.stack([sample[variable] for sample in outputs_per_sample])

    return outputs


def call_sample_fn(sample_fn, arguments):
    parameters = inspect.signature(sample_fn).parameters
    required_arguments = {name: arguments[name] for name in parameters if name in arguments}

    return sample_fn(**required_arguments)


def draw_num_samples(reps, meta, combinations):
    if callable(reps):
        return call_sample_fn(reps, meta | {"size": len(combinations), "combinations": combinations})
    if isinstance(reps, str):
        return meta[reps]

    return reps
