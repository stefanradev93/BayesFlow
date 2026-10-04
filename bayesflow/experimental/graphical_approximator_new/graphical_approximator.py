from functools import partial

import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.approximators import Approximator
from bayesflow.networks import InferenceNetwork
from bayesflow.utils.serialization import serializable, serialize

from ..graphical_simulator_new.samples import NodeSamples, Simulation, placeholder_simulation, repeat
from ..graphs_new import InvertedGraph
from ..graphs_new.simulation_graph import data_nodes
from .batches import SimulationDataset, groupings, to_batch
from .conditions import network_conditions, network_variables, split_values
from .encoder import ExchangeableEncoder


@serializable("bayesflow.experimental")
class GraphicalApproximator(Approximator):
    def __init__(
        self,
        graph: InvertedGraph,
        *,
        inference_networks: list[InferenceNetwork],
        encoders: list[dict[str, keras.Layer]] | None = None,
        adapter: Adapter | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.graph = graph
        self.inference_networks = inference_networks
        self.adapter = adapter or Adapter()

        simulation_graph = graph.expanded_graph.simulation_graph
        self.data_nodes = data_nodes(simulation_graph)
        self.networks = graph.networks()

        self.groupings = {}

        for node in self.data_nodes:
            self.groupings[node] = groupings(simulation_graph, node)

        self.conditions = []

        for network in self.networks:
            if not graph.is_amortizable(network[0]):
                raise NotImplementedError(f"Network for {network} is not amortizable.")

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

        batch = to_batch(repeat(observed, num_samples), self.adapter, self.groupings)

        for i, network in enumerate(self.networks):
            network_conds = network_conditions(batch, network, self.conditions[i], self.data_nodes, self.encoders[i])
            samples = self.inference_networks[i].sample((keras.ops.shape(network_conds)[0],), conditions=network_conds)

            for node, values in split_values(samples, network, self.variable_dims).items():
                batch[node]["values"] = values

        result = {}

        for network in self.networks:
            for node in network:
                for variable, value in batch[node]["values"].items():
                    value = keras.ops.convert_to_numpy(value)
                    value = value.reshape(num_samples, -1, value.shape[-1])

                    result[variable] = np.moveaxis(value, 0, 1)

        result = self.adapter(result, inverse=True, strict=False)

        return keras.tree.map_structure(keras.ops.convert_to_numpy, result)

    def build_dataset(self, *, simulator, batch_size, num_batches):
        batch_fn = partial(to_batch, adapter=self.adapter, groupings=self.groupings)

        return SimulationDataset(simulator, batch_size, batch_fn, num_batches)

    def compute_metrics(self, stage: str = "training", **batch):
        total_loss = 0
        metrics = {}

        for i, network in enumerate(self.networks):
            variables = network_variables(batch, network)
            conditions = network_conditions(
                batch, network, self.conditions[i], self.data_nodes, self.encoders[i], training=stage == "training"
            )

            network_metrics = self.inference_networks[i].compute_metrics(variables, conditions=conditions, stage=stage)
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
        }

        return super().get_config() | serialize(config)

    def get_build_config(self):
        return {"variable_dims": self.variable_dims}

    def build_from_config(self, config):
        self.variable_dims = config["variable_dims"]

        simulation_graph = self.graph.expanded_graph.simulation_graph
        simulation = placeholder_simulation(simulation_graph, self.variable_dims)

        self.compute_metrics(**to_batch(simulation, Adapter(), self.groupings), stage="training")
        self.built = True

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
