import keras
import networkx as nx
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.approximators import Approximator
from bayesflow.networks import InferenceNetwork
from bayesflow.networks.helpers.standardization.standardization import Standardization
from bayesflow.utils.serialization import serializable, serialize

from ..graphical_simulator_new.samples import (
    NodeSamples,
    Simulation,
    concatenate,
    num_datasets,
    num_rows,
)
from ..graphs_new import InvertedGraph
from ..graphs_new.simulation_graph import data_nodes
from .batches import SimulationDataset, groupings, to_batch
from .conditions import network_conditions, network_variables, sequential_conditions, split_values
from .encoder import ExchangeableEncoder, sum_per_dataset


@serializable("bayesflow.experimental")
class GraphicalApproximator(Approximator):
    def __init__(
        self,
        graph: InvertedGraph,
        *,
        inference_networks: list[InferenceNetwork],
        encoders: list[dict[str, keras.Layer]] | None = None,
        adapter: Adapter | None = None,
        standardize: str | list[str] | None = "inference_variables",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.graph = graph
        self.inference_networks = inference_networks
        self.adapter = adapter or Adapter()
        self.standardize = standardize

        simulation_graph = graph.expanded_graph.simulation_graph
        self.data_nodes = data_nodes(simulation_graph)
        self.networks = graph.networks()

        self.groupings = {}

        for node in self.data_nodes:
            self.groupings[node] = groupings(simulation_graph, node)

        self.conditions = []
        self.sequential = []

        for network in self.networks:
            if not graph.is_amortizable(network[0]):
                if len(network) > 1:
                    raise NotImplementedError(f"Network for {network} infers several non-amortizable nodes.")

                self.sequential.append(network[0])

            self.conditions.append(graph.conditions(network))

        if encoders is None:
            encoders = []

            for conditions in self.conditions:
                network_encoders = {}

                for node in conditions:
                    if node in self.data_nodes:
                        network_encoders[node] = ExchangeableEncoder()

                encoders.append(network_encoders)

        self.encoders = encoders

    def sample(self, *, num_samples: int, conditions: Simulation) -> dict[str, np.ndarray]:
        observed = {}

        for node, samples in conditions.items():
            if node in self.data_nodes:
                observed[node] = samples
            else:
                observed[node] = NodeSamples(samples.indices, {})

        batch = self.to_batch(concatenate([observed] * num_samples))
        batch = self.standardized(batch, stage="inference")

        for i, network in enumerate(self.networks):
            if network[0] in self.sequential:
                self.sample_sequentially(batch, i)
                continue

            conditions = self.inference_conditions(batch, i)
            samples = self.inference_networks[i].sample((keras.ops.shape(conditions)[0],), conditions=conditions)

            for node, values in split_values(samples, network, self.variable_dims).items():
                batch[node]["values"] = values

        result = {}

        for network in self.networks:
            for node in network:
                for variable, value in batch[node]["values"].items():
                    value = self.standardizer.maybe_standardize(value, variable, forward=False)
                    value = keras.ops.convert_to_numpy(value)
                    value = value.reshape(num_samples, -1, value.shape[-1])

                    result[variable] = np.moveaxis(value, 0, 1)

        result = self.adapter(result, inverse=True, strict=False)

        return keras.tree.map_structure(keras.ops.convert_to_numpy, result)

    def log_prob(self, data: Simulation) -> np.ndarray:
        batch = self.standardized(self.to_batch(data), stage="inference")
        datasets = num_datasets(data)

        log_prob = keras.ops.zeros(datasets)

        for i, network in enumerate(self.networks):
            variables = network_variables(batch, network)
            conditions = self.inference_conditions(batch, i)

            network_log_prob = self.inference_networks[i].log_prob(variables, conditions=conditions)
            log_prob += sum_per_dataset(network_log_prob, batch[network[0]]["indices"]["dataset"], datasets)

            for node in network:
                for variable, value in batch[node]["values"].items():
                    if variable in self.standardizer.standardize:
                        _, log_det_jac = self.standardizer.maybe_standardize(value, variable, log_det_jac=True)
                        log_prob += sum_per_dataset(log_det_jac, batch[node]["indices"]["dataset"], datasets)

                _, log_det_jac = self.adapter(data[node].values, strict=False, log_det_jac=True)

                for value in log_det_jac.values():
                    value = keras.ops.cast(value, keras.config.floatx())
                    log_prob += sum_per_dataset(value, batch[node]["indices"]["dataset"], datasets)

        return keras.ops.convert_to_numpy(log_prob)

    def build_dataset(self, *, simulator, batch_size, num_batches):
        return SimulationDataset(simulator, batch_size, self.to_batch, num_batches)

    def fit(self, *, dataset: Simulation | keras.utils.PyDataset | None = None, **kwargs):
        if isinstance(dataset, dict):
            dataset = SimulationDataset(dataset, kwargs.pop("batch_size"), self.to_batch)

        return super().fit(dataset=dataset, **kwargs)

    def compute_metrics(self, stage: str = "training", **batch):
        total_loss = 0
        metrics = {}

        if not self.built:
            self.build_from_data(batch)

        batch = self.standardized(batch, stage)

        for i, network in enumerate(self.networks):
            variables = network_variables(batch, network)
            conditions = self.inference_conditions(batch, i, training=stage == "training")
            weights = batch[network[0]].get("weights")

            network_metrics = self.inference_networks[i].compute_metrics(
                variables, conditions=conditions, sample_weight=weights, stage=stage
            )
            total_loss += network_metrics["loss"]

            for name, value in network_metrics.items():
                metrics[f"network_{i}/{name}"] = value

        metrics["loss"] = total_loss

        return metrics

    def get_config(self):
        config = {
            "graph": self.graph,
            "inference_networks": self.inference_networks,
            "encoders": self.encoders,
            "adapter": self.adapter,
            "standardize": self.standardize,
        }

        return super().get_config() | serialize(config)

    def get_build_config(self):
        return {"variable_dims": self.variable_dims}

    def build_from_config(self, config):
        self.variable_dims = config["variable_dims"]

        variable_shapes = {}
        standardized_variables = []

        for node, dims in self.variable_dims.items():
            for variable, dim in dims.items():
                variable_shapes[variable] = (None, dim)

                if self.standardize == "all":
                    standardized_variables.append(variable)
                elif self.standardize == "inference_variables":
                    if node not in self.data_nodes:
                        standardized_variables.append(variable)
                elif variable in (self.standardize or []):
                    standardized_variables.append(variable)

        self.standardizer = Standardization(standardized_variables)
        self.standardizer.build(variable_shapes)

        simulation_graph = self.graph.expanded_graph.simulation_graph
        simulation = placeholder_simulation(simulation_graph, self.variable_dims)

        self.built = True

        self.compute_metrics(**self.to_batch(simulation, adapter=Adapter()), stage="validation")

    def to_batch(self, simulation: Simulation, adapter: Adapter | None = None, padded: bool = False) -> dict:
        datasets = num_datasets(simulation)

        # the nodes of a network are concatenated row by row, so each must have one row per dataset
        for network in self.networks:
            if len(network) > 1:
                for node in network:
                    if num_rows(simulation[node].indices) != datasets:
                        raise ValueError(
                            f"Network {network} infers {node}, which has more than one row per dataset. "
                            "The nodes of a network must share their rows."
                        )

        return to_batch(simulation, adapter or self.adapter, self.groupings, self.sequential, padded)

    def inference_conditions(self, batch: dict, i: int, training: bool = False):
        network = self.networks[i]
        conditions = network_conditions(
            batch, network, self.conditions[i], self.data_nodes, self.encoders[i], training=training
        )

        if network[0] in self.sequential:
            conditions = sequential_conditions(batch, network[0], conditions)

        return conditions

    def sample_sequentially(self, batch: dict, i: int):
        node = self.networks[i][0]
        positions = batch[node]["steps"]["positions"]

        values = {}

        for variable, dim in self.variable_dims[node].items():
            values[variable] = keras.ops.zeros((len(positions), dim))

        batch[node]["values"] = values

        for step in range(int(np.max(positions)) + 1):
            conditions = self.inference_conditions(batch, i)
            samples = self.inference_networks[i].sample((len(positions),), conditions=conditions)
            is_step = (positions == step)[:, None]

            for variable, value in split_values(samples, [node], self.variable_dims)[node].items():
                values[variable] = keras.ops.where(is_step, value, values[variable])

    def standardized(self, batch: dict, stage: str) -> dict:
        result = {}

        for node, entries in batch.items():
            values = {}

            for variable, value in entries["values"].items():
                values[variable] = self.standardizer.maybe_standardize(value, variable, stage=stage)

            result[node] = entries | {"values": values}

        return result

    def build_from_data(self, batch):
        variable_dims = {}

        for node, samples in batch.items():
            variable_dims[node] = {}

            for variable, value in batch[node]["values"].items():
                variable_dims[node][variable] = int(keras.ops.shape(value)[-1])

        self.build_from_config({"variable_dims": variable_dims})

    def _batch_size_from_data(self, data):
        root = self.networks[0][0]

        return keras.ops.shape(data[root]["indices"][root])[0]


def placeholder_simulation(graph, variable_dims: dict[str, dict[str, int]]) -> Simulation:
    simulation = {}

    for node in graph:
        indices = {}
        for column in ["dataset", *nx.ancestors(graph, node), node]:
            indices[column] = np.zeros(1, dtype=np.int64)

        values = {}

        for variable, dim in variable_dims[node].items():
            values[variable] = np.zeros((1, dim))

        simulation[node] = NodeSamples(indices, values)

    return simulation
