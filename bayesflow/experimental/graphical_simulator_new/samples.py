from dataclasses import dataclass
from typing import TypeAlias

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


def num_datasets(simulation: Simulation) -> int:
    return 1 + max(samples.indices["dataset"].max() for samples in simulation.values())
