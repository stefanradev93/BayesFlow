from dataclasses import dataclass
from typing import TypeAlias

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class NodeSamples:
    indices: pd.DataFrame
    values: dict[str, np.ndarray]


Simulation: TypeAlias = dict[str, NodeSamples]


def concatenate(simulations: list[Simulation]) -> Simulation:
    rows = []
    for simulation in simulations:
        sizes = {"dataset": num_datasets(simulation)}

        for node, samples in simulation.items():
            sizes[node] = len(samples.indices)

        rows.append(sizes)

    sizes = pd.DataFrame(rows)

    offsets = sizes.cumsum() - sizes

    concatenated = {}
    for node, first in simulations[0].items():
        columns = first.indices.columns

        indices = []
        for i, simulation in enumerate(simulations):
            indices.append(simulation[node].indices + offsets.loc[i, columns])

        values = {}
        for variable in first.values:
            parts = []
            for simulation in simulations:
                parts.append(simulation[node].values[variable])

            values[variable] = np.concatenate(parts)

        concatenated[node] = NodeSamples(pd.concat(indices, ignore_index=True), values)

    return concatenated


def split(simulation: Simulation) -> list[Simulation]:
    positions = {}

    for node, samples in simulation.items():
        positions[node] = samples.indices.groupby("dataset").cumcount().to_numpy()

    datasets = []

    for dataset in range(num_datasets(simulation)):
        datasets.append({})

    for node, samples in simulation.items():
        indices = samples.indices.copy()

        for column in indices.columns.drop("dataset"):
            indices[column] = positions[column][indices[column]]

        indices["dataset"] = 0

        for dataset, rows in samples.indices.groupby("dataset").indices.items():
            values = {}

            for variable, value in samples.values.items():
                values[variable] = value[rows]

            datasets[dataset][node] = NodeSamples(indices.iloc[rows].reset_index(drop=True), values)

    return datasets


def repeat(simulation, num_copies: int) -> Simulation:
    return concatenate([simulation] * num_copies)


def placeholder_simulation(graph, variable_dims: dict[str, dict[str, int]]) -> Simulation:
    simulation = {}

    for node in graph:
        columns = ["dataset", *nx.ancestors(graph, node), node]
        indices = pd.DataFrame(0, index=[0], columns=columns)

        values = {}

        for variable, dim in variable_dims[node].items():
            values[variable] = np.zeros((1, dim))

        simulation[node] = NodeSamples(indices, values)

    return simulation


def num_datasets(simulation: Simulation) -> int:
    return 1 + max(samples.indices["dataset"].max() for samples in simulation.values())
