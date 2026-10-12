from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .indexing_helpers import concatenated_ranges


@dataclass
class NodeSamples:
    """
    The samples of one node in long format.

    Every row holds one sample of the node, and the rows are sorted by dataset.

    Attributes
    ----------
    indices : dict[str, np.ndarray]
        Integer index columns of shape `(num_rows,)`: ``"dataset"`` with the
        dataset of each row, one column per ancestor with the row of that
        ancestor the sample belongsto, and one column named after the node
        with the row's own position. Ids count rows of the whole simulation,
        not within each dataset.
    values : dict[str, np.ndarray]
        The samples of each variable of the node, with shape `(num_rows, dim)`.
    """

    indices: dict[str, np.ndarray]
    values: dict[str, np.ndarray]


#: The samples of every node of a GraphicalSimulator, keyed by node name.
Simulation: TypeAlias = dict[str, NodeSamples]


def num_rows(indices: dict[str, np.ndarray]) -> int:
    """
    Counts the rows of a node.

    Parameters
    ----------
    indices : dict[str, np.ndarray]
        The index columns of the node, as in :py:attr:`NodeSamples.indices`

    Returns
    -------
    int
        The number of rows.
    """
    return len(indices["dataset"])


def num_datasets(simulation: Simulation) -> int:
    """
    Counts the datasets in a simulation.

    Parameters
    ----------
    simulation : Simulation
        The samples of every node.

    Returns
    -------
    int
        The number of datasets.
    """
    return 1 + max(int(np.max(samples.indices["dataset"])) for samples in simulation.values())


def concatenate(simulations: list[Simulation]) -> Simulation:
    """
    Combines several simulations into one.

    The datasets of later simulations are numbered after those of earlier ones.
    This also shifts the index columns, so that every index is consecutive.

    Parameters
    ----------
    simulation : list[Simulation]
        Simulations of the same graph, with the same nodes and variables.

    Returns
    -------
    Simulation
        A simulation holding the datasets of all simulations.
    """
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
    """
    Keeps only some datasets of a simulation.

    The selected datasets are renumbered from 0 in increasing order of their
    original index, and the index columns are renumbered to match.

    Parameters
    ----------
    simulation : Simulation
        The samples of every node.
    datasets : np.ndarray
        The numbers of the datasets to keep.

    Returns
    -------
    Simulation
        A simulation holding only the selected datasets.
    """
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
