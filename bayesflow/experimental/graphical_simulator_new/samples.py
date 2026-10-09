from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .indexing_helpers import concatenated_ranges


@dataclass
class NodeSamples:
    # rows are sorted by dataset
    indices: dict[str, np.ndarray]
    values: dict[str, np.ndarray]


Simulation: TypeAlias = dict[str, NodeSamples]


def num_rows(indices: dict[str, np.ndarray]) -> int:
    return len(indices["dataset"])


def num_datasets(simulation: Simulation) -> int:
    return 1 + max(int(np.max(samples.indices["dataset"])) for samples in simulation.values())


def concatenate(simulations: list[Simulation]) -> Simulation:
    # Concatenate simulations by shifting all indices past those of the previous simulations.

    concatenated = {}

    for node, first in simulations[0].items():
        indices = {}

        for column in first.indices:
            parts = []
            offset = 0
            for simulation in simulations:
                parts.append(simulation[node].indices[column] + offset)

                if column == "dataset":
                    offset += num_datasets(simulation)
                else:
                    offset += num_rows(simulation[column].indices)

            indices[column] = np.concatenate(parts)

        values = {}

        for variable in first.values:
            parts = []
            for simulation in simulations:
                parts.append(simulation[node].values[variable])

            values[variable] = np.concatenate(parts)

        concatenated[node] = NodeSamples(indices, values)

    return concatenated


def select(simulation: Simulation, datasets: np.ndarray) -> Simulation:
    # select only indices and values with a specific dataset ID

    datasets = np.sort(datasets)

    rows = {"dataset": datasets}
    for node, samples in simulation.items():
        starts = np.searchsorted(samples.indices["dataset"], datasets, side="left")
        ends = np.searchsorted(samples.indices["dataset"], datasets, side="right")
        rows[node] = concatenated_ranges(starts, ends - starts)

    selected = {}
    for node, samples in simulation.items():
        indices = {}
        for column, ids in samples.indices.items():
            # renumbers ids by their position among the selected rows
            indices[column] = np.searchsorted(rows[column], ids[rows[node]])

        values = {}
        for variable, value in samples.values.items():
            values[variable] = value[rows[node]]

        selected[node] = NodeSamples(indices, values)

    return selected
