import itertools
import math
from typing import Callable

import keras
import networkx as nx
import numpy as np

from bayesflow.adapters import Adapter

from ..graphical_simulator_new import GraphicalSimulator
from ..graphical_simulator_new.indexing_helpers import concatenated_ranges, group_ids
from ..graphical_simulator_new.samples import Simulation, num_datasets, select
from ..graphs_new import InvertedGraph, SimulationGraph
from ..graphs_new.simulation_graph import is_exchangeable, is_interior


class SimulationDataset(keras.utils.PyDataset):
    def __init__(
        self,
        source: GraphicalSimulator | Simulation,
        batch_size: int,
        to_batch: Callable[[Simulation], dict],
        num_batches: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source = source
        self.batch_size = batch_size
        self.to_batch = to_batch

        if isinstance(source, GraphicalSimulator):
            self.length = num_batches
        else:
            self.length = math.ceil(num_datasets(source) / batch_size)
            self.order = np.random.permutation(num_datasets(source))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> dict:
        if isinstance(self.source, GraphicalSimulator):
            return self.to_batch(self.source.sample(self.batch_size), padded=True)

        datasets = self.order[i * self.batch_size : (i + 1) * self.batch_size]

        return self.to_batch(select(self.source, datasets), padded=True)

    def on_epoch_end(self) -> None:
        if not isinstance(self.source, GraphicalSimulator):
            self.order = np.random.permutation(num_datasets(self.source))


def to_batch(simulation: Simulation, graph: InvertedGraph, adapter: Adapter, padded: bool = False) -> dict:
    # Adds groups and steps to the samples and runs the adapter.
    # This is the input that inference networks use
    simulation_graph = graph.expanded_graph.simulation_graph

    batch = {}
    for node, samples in simulation.items():
        batch[node] = {"indices": samples.indices, "values": samples.values}

        if simulation_graph.out_degree(node) == 0:
            batch[node]["groups"] = groups(simulation_graph, node, samples.indices)
        elif not graph.is_amortizable(node):
            batch[node]["steps"] = steps(node, samples.indices)

    if padded:
        batch = pad(batch)

    for entries in batch.values():
        entries["values"] = adapter(entries["values"], strict=False)

    return batch


def groups(graph: SimulationGraph, data_node: str, indices: dict[str, np.ndarray]) -> dict[str, dict]:
    # Helper function for to_batch.
    # For each grouping of exchangeable factors, compute to which group the row belongs to.
    result = {}

    for factors in groupings(graph, data_node):
        columns = []
        for factor in ["dataset", *factors]:
            columns.append(indices[factor])

        ids = group_ids(columns)
        name = "+".join(factors) or "dataset"
        result[name] = {"ids": ids, "counts": np.bincount(ids)}

    return result


def groupings(graph: SimulationGraph, data_node: str) -> list[tuple[str, ...]]:
    # Lists the different ways that rows of the data node can be grouped by.
    # This is necessary for summarizing the data.
    ancestors = nx.ancestors(graph, data_node)

    grouping_factors = []
    for node in nx.lexicographical_topological_sort(graph):
        if node in ancestors and is_interior(graph, node) and is_exchangeable(graph, node):
            grouping_factors.append(node)

    valid = []
    for size in range(len(grouping_factors) + 1):
        for factors in itertools.combinations(grouping_factors, size):
            outer_factors = set()
            for factor in factors:
                outer_factors |= nx.ancestors(graph, factor) & set(grouping_factors)

            if outer_factors <= set(factors):
                valid.append(factors)

    if not is_exchangeable(graph, data_node):
        valid.remove(tuple(grouping_factors))

    return valid


def steps(node: str, indices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Helper function for to_batch.
    # Computes for each row its sequence and position, as well as all (earlier, later)
    # row pairs within a sequence. Used for sequentially inferred nodes.
    columns = []
    for column, ids in indices.items():
        if column != node:
            columns.append(ids)

    ids = group_ids(columns)
    counts = np.bincount(ids)
    starts = np.cumsum(counts) - counts
    order = np.argsort(ids, kind="stable")

    positions = np.empty(len(ids), dtype=np.int64)
    positions[order] = np.arange(len(ids)) - np.repeat(starts, counts)

    later = np.repeat(np.arange(len(ids)), positions)
    earlier = order[concatenated_ranges(starts[ids], positions)]

    return {"ids": ids, "counts": counts, "positions": positions, "earlier": earlier, "later": later}


def pad(batch: dict) -> dict:
    # Pad output of to_batch, so that inference networks do not see so many
    # dataset sizes, which would trigger a long compilation run
    sizes = {}
    datasets = 0

    for node, entries in batch.items():
        sizes[node] = bucket(len(entries["indices"][node]))
        datasets = max(datasets, int(np.max(entries["indices"]["dataset"])) + 1)

    sizes["dataset"] = bucket(datasets)

    for node, entries in batch.items():
        rows = len(entries["indices"][node])
        size = sizes[node]

        indices = {}
        for column, ids in entries["indices"].items():
            indices[column] = pad_with(ids, size, sizes[column] - 1)

        values = {}
        for variable, value in entries["values"].items():
            values[variable] = repeat_rows(value, size)

        weights = np.full(rows, size / rows, dtype=keras.config.floatx())

        entries["indices"] = indices
        entries["values"] = values
        entries["weights"] = pad_with(weights, size, 0.0)

        groupings = list(entries.get("groups", {}).values())
        if "steps" in entries:
            groupings.append(entries["steps"])

        for group in groupings:
            num_groups = bucket(len(group["counts"]))
            group["ids"] = pad_with(group["ids"], size, num_groups - 1)
            group["counts"] = pad_with(group["counts"], num_groups, 0)

        if "steps" in entries:
            steps = entries["steps"]
            num_pairs = bucket(len(steps["earlier"]))
            steps["positions"] = pad_with(steps["positions"], size, 0)
            steps["earlier"] = pad_with(steps["earlier"], num_pairs, size - 1)
            steps["later"] = pad_with(steps["later"], num_pairs, size - 1)

    return batch


def bucket(n: int) -> int:
    # the next of four sizes per doubling that is larger than n
    return int(np.ceil(2 ** (np.ceil(4 * np.log2(n + 1)) / 4)))


def pad_with(array, size: int, value):
    padding = np.full((size - len(array), *np.shape(array)[1:]), value, dtype=np.asarray(array).dtype)

    return np.concatenate([array, padding])


def repeat_rows(values, size: int):
    return np.asarray(values)[np.arange(size) % len(values)]
