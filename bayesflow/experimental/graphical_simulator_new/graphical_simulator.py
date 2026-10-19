import inspect
from typing import Any, Callable

import networkx as nx
import numpy as np

from ..graphs_new import SimulationGraph
from .indexing_helpers import join
from .samples import NodeSamples, Simulation, num_rows


class GraphicalSimulator:
    """
    A graph-based simulator for multilevel models.

    The model is a directed acyclic graph whose nodes hold sampling functions.
    Every node creates samples once for each combination of its parent's samples.
    The data are stored in long format: one row per sample, with integer indices
    of the dataset, all ancestors, and the node itself.

    Parameters
    ----------
    meta_fn: Callable([], dict[str, Any]], optional
        Function returning a dict of meta data, called once per dataset in ``sample``,
        so that every dataset has its own meta data.
        Meta data can set the number of repetitions of a node via ``reps``,
        and sampling functions can accept all keys of the return meta data.
    is_batched: bool, optional
        Default value for ``is_batched`` for ``add_node``. If `True`, all added
        nodes are batched by default, if `False`, all added nodes are unbatched.
        Default is False.
    """

    def __init__(self, meta_fn: Callable[[], dict[str, Any]] | None = None, is_batched: bool = False):
        self.graph = SimulationGraph(meta_fn=meta_fn)
        self.is_batched = is_batched

    def add_node(
        self,
        node: str,
        sample_fn: Callable[..., dict[str, Any]],
        reps: int | str | Callable = 1,
        is_batched: bool | None = None,
    ):
        """
        Adds a node with its sampling function and number of repetitions.

        Parameters
        ----------
        node : str
            Name of the node.
        sample_fn: Callable[..., dict[str, Any]]
            Function returning a dict that maps variable names to sampled values.
            Each ``sample_fn`` of a node can accept arguments that are named
            by either the variable names of the node's parents or the meta data.
        reps : int, str or Callable, optional
            Number of repetitions of the node per combination of its parents'
            values. Used to vary the number of groups for each simulated dataset.
            Either a fixed integer, a key of the meta data, or a function that takes
            a size, and optionally a combinations argument, or keys of the
            meta data, and returns the number of repetitions for
            each of the parent combinations. Default is 1.
            If ``reps`` is a function, it must be defined at the top-level,
            otherwise the GraphicalApproximator cannot be saved.
        is_batched : bool, optional
            If False, ``sample_fn`` is called once per sample with the values
            of its parents. For much faster batched sampling, set it to `True`.
            In this case, ``sample_fn`` must take an addition argument called
            ``batch_size`` and return a dictionary of sampled values with shape
            (batch_size, dim). The ``is_batched`` argument in GraphicalSimulator
            can be used to set the default for all nodes.
        """
        if is_batched is None:
            is_batched = self.is_batched

        self.graph.add_node(node, sample_fn=sample_fn, reps=reps, is_batched=is_batched)

    def add_edge(self, from_node: str, to_node: str):
        """
        Adds a directed edge indicating a dependency between two nodes.

        Parameters
        ----------
        from_node : str
            Parent node, whose variables are passed to the sampling function of
            the child.
        to_node : str
            Child node, which is repeated for every sample of the parent.
        """
        self.graph.add_edge(from_node, to_node)

    def sample(self, num_datasets: int, meta: dict[str, Any] | None = None) -> Simulation:
        """
        Samples datasets by traversing the graph in topological order.

        Parameters
        ----------
        num_datasets : int
            The number of datasets to sample.
        meta : dict[str, Any], optional
            Meta data shared by all datasets, overriding the output of ``meta_fn``.

        Returns
        -------
        simulation : dict[str, NodeSamples]
            The samples of every node of type ``NodeSamples``.
            ``NodeSample`` contains two fields: ``indices`` contains a dictionary
            mapping each sample to its dataset, every ancestor and the node index
            itself. ``values`` maps every variable to an array of shape
            ``(num_samples, dim)``. The output can be used as input to
            ``GraphicalApproximator.fit``.
        """
        meta = meta or {}
        dataset_meta = sample_dataset_meta(self.graph.meta_fn, num_datasets)
        for key in meta:
            dataset_meta.pop(key, None)

        simulation = {}

        for node in nx.lexicographical_topological_sort(self.graph):
            parents = {}
            for parent in self.graph.predecessors(node):
                parents[parent] = simulation[parent]

            settings = self.graph.nodes[node]
            indices = assign_indices(node, parents, settings["reps"], meta, dataset_meta, num_datasets)
            values = draw_values(settings["sample_fn"], parents, indices, meta, dataset_meta, settings["is_batched"])

            simulation[node] = NodeSamples(indices, values)

        return simulation


def sample_dataset_meta(meta_fn, num_datasets) -> dict[str, np.ndarray]:
    # one draw of meta_fn per dataset, stacked into one array per key
    if meta_fn is None or num_datasets == 0:
        return {}

    draws = [meta_fn() for _ in range(num_datasets)]

    return {k: np.stack([np.asarray(draw[k]) for draw in draws]) for k in draws[0]}


def assign_indices(node, parents, reps, meta, dataset_meta, num_datasets) -> dict[str, np.ndarray]:
    combinations = {"dataset": np.arange(num_datasets)}

    for samples in parents.values():
        combinations = join(combinations, samples.indices)

    combination_meta = {k: v[combinations["dataset"]] for k, v in dataset_meta.items()}
    meta = meta | combination_meta

    if callable(reps):
        repetitions = call(reps, meta | {"size": num_rows(combinations), "combinations": combinations})
    elif isinstance(reps, str):
        repetitions = meta[reps]
    else:
        repetitions = reps

    indices = {}
    for column, ids in combinations.items():
        indices[column] = np.repeat(ids, repetitions)

    indices[node] = np.arange(num_rows(indices))

    return indices


def draw_values(sample_fn, parents, indices, meta, dataset_meta, is_batched) -> dict[str, np.ndarray]:
    rows = num_rows(indices)
    row_meta = {k: v[indices["dataset"]] for k, v in dataset_meta.items()}

    parent_values = {}
    for parent, samples in parents.items():
        for variable, values in samples.values.items():
            parent_values[variable] = values[indices[parent]]

    if is_batched:
        outputs = call(sample_fn, parent_values | meta | row_meta | {"batch_size": rows, "indices": indices})
    else:
        outputs = draw_row_by_row(sample_fn, parent_values | row_meta, meta, rows)

    values = {}
    for variable, value in outputs.items():
        values[variable] = np.reshape(value, (rows, -1))

    return values


def draw_row_by_row(sample_fn, parent_values, meta, rows) -> dict[str, np.ndarray]:
    draws = []

    for row in range(rows):
        arguments = {}
        for variable, values in parent_values.items():
            arguments[variable] = np.squeeze(values[row])

        draws.append(call(sample_fn, arguments | meta))

    outputs = {}
    for variable in draws[0]:
        outputs[variable] = np.stack([draw[variable] for draw in draws])

    return outputs


def call(function, arguments):
    # calls the function with the arguments that appear in its signature
    accepted = {}
    for name in inspect.signature(function).parameters:
        if name in arguments:
            accepted[name] = arguments[name]

    return function(**accepted)
