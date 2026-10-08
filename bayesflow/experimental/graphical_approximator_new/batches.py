import itertools
import math
from typing import Callable

import keras
import networkx as nx
import numpy as np

from bayesflow.adapters import Adapter

from ..graphical_simulator_new import GraphicalSimulator
from ..graphical_simulator_new.samples import Simulation, concatenate, split
from ..graphs_new import SimulationGraph
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
            self.source = split(source)
            self.length = math.ceil(len(self.source) / batch_size)
            self.order = np.random.permutation(len(self.source))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> dict:
        if isinstance(self.source, GraphicalSimulator):
            return pad(self.to_batch(self.source.sample(self.batch_size)))

        simulations = []

        for j in self.order[i * self.batch_size : (i + 1) * self.batch_size]:
            simulations.append(self.source[j])

        return pad(self.to_batch(concatenate(simulations)))

    def on_epoch_end(self) -> None:
        if not isinstance(self.source, GraphicalSimulator):
            self.order = np.random.permutation(len(self.source))


def to_batch(
    simulation: Simulation, adapter: Adapter, groupings: dict[str, list[tuple[str, ...]]], sequential: list[str] = ()
) -> dict:
    batch = {}

    for node, samples in simulation.items():
        indices = {column: ids.to_numpy() for column, ids in samples.indices.items()}
        batch[node] = {"indices": indices, "values": adapter(samples.values, strict=False)}

    for data_node, node_groupings in groupings.items():
        groups = {}

        for factors in node_groupings:
            ids = simulation[data_node].indices.groupby(["dataset", *factors], sort=False).ngroup().to_numpy()
            groups["+".join(factors) or "dataset"] = {"ids": ids, "counts": np.bincount(ids)}

        batch[data_node]["groups"] = groups

    for node in sequential:
        indices = simulation[node].indices
        group = indices.groupby(list(indices.columns.drop(node)), sort=False)

        earlier = []
        later = []

        for rows in group.indices.values():
            first, second = np.triu_indices(len(rows), k=1)
            earlier.append(rows[first])
            later.append(rows[second])

        batch[node]["steps"] = {
            "ids": group.ngroup().to_numpy(),
            "counts": group.size().to_numpy(),
            "positions": group.cumcount().to_numpy(),
            "earlier": np.concatenate(earlier),
            "later": np.concatenate(later),
        }

    return batch


def pad(batch: dict) -> dict:
    sizes = {}

    for node, entries in batch.items():
        sizes[node] = bucket(len(entries["indices"][node]))

    datasets = 0
    for entries in batch.values():
        datasets = max(datasets, int(np.max(entries["indices"]["dataset"])) + 1)

    sizes["dataset"] = bucket(datasets)

    padded = {}

    for node, entries in batch.items():
        rows = len(entries["indices"][node])
        size = sizes[node]

        indices = {}
        for column, ids in entries["indices"].items():
            indices[column] = pad_with(ids, size, sizes[column] - 1)

        values = {}
        for variable, value in entries["values"].items():
            values[variable] = repeat_rows(value, size)

        weights = pad_with(np.full(rows, size / rows, dtype=keras.config.floatx()), size, 0.0)

        padded[node] = {"indices": indices, "values": values, "weights": weights}

        if "groups" in entries:
            groups = {}

            for name, group in entries["groups"].items():
                num_groups = bucket(len(group["counts"]))
                groups[name] = {
                    "ids": pad_with(group["ids"], size, num_groups - 1),
                    "counts": pad_with(group["counts"], num_groups, 0),
                }

            padded[node]["groups"] = groups

        if "steps" in entries:
            steps = entries["steps"]
            num_groups = bucket(len(steps["counts"]))
            num_pairs = bucket(len(steps["earlier"]))

            padded[node]["steps"] = {
                "ids": pad_with(steps["ids"], size, num_groups - 1),
                "counts": pad_with(steps["counts"], num_groups, 0),
                "positions": pad_with(steps["positions"], size, 0),
                "earlier": pad_with(steps["earlier"], num_pairs, size - 1),
                "later": pad_with(steps["later"], num_pairs, size - 1),
            }

    return padded


def bucket(n: int) -> int:
    return int(np.ceil(2 ** (np.ceil(4 * np.log2(n + 1)) / 4)))


def pad_with(array, size: int, value):
    padding = np.full((size - len(array), *np.shape(array)[1:]), value, dtype=np.asarray(array).dtype)

    return np.concatenate([array, padding])


def repeat_rows(values, size: int):
    return np.asarray(values)[np.arange(size) % len(values)]


def groupings(graph: SimulationGraph, data_node: str) -> list[tuple[str, ...]]:
    grouping_factors = []

    for node in nx.lexicographical_topological_sort(graph):
        interior = is_interior(graph, node)
        exchangeable = is_exchangeable(graph, node)

        if node in nx.ancestors(graph, data_node) and interior and exchangeable:
            grouping_factors.append(node)

    result = []

    for size in range(len(grouping_factors) + 1):
        for factors in itertools.combinations(grouping_factors, size):
            outer_factors = set()

            for factor in factors:
                outer_factors |= nx.ancestors(graph, factor) & set(grouping_factors)

            if outer_factors <= set(factors):
                result.append(factors)

    if not is_exchangeable(graph, data_node):
        result.remove(tuple(grouping_factors))

    return result
