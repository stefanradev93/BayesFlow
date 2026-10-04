import keras

from .encoder import gather, segment_count, segment_mean


def network_conditions(
    batch: dict,
    network: list[str],
    conditions: list[str],
    data_nodes: list[str],
    encoders: dict[str, keras.Layer],
    training=False,
):
    target = network[0]
    target_indices = batch[target]["indices"]
    num_instances = keras.ops.shape(target_indices[target])[0]

    latent = []
    observed = []

    for node in conditions:
        if node in data_nodes:
            observed.append(node)
        else:
            latent.append(node)

    result = []

    for data_node in observed:
        observations = batch[data_node]["indices"]

        if target not in observations:
            raise NotImplementedError(f"{target} conditions on {data_node}")

        features = [node_values(batch, data_node)]
        for node in latent:
            if node in observations:
                features.append(gather(node_values(batch, node), observations[node]))

        groups = []
        for name in sorted(batch[data_node]["groups"]):
            groups.append(batch[data_node]["groups"][name])

        encoded = encoders[data_node](keras.ops.concatenate(features, axis=-1), groups, training=training)

        pooled = segment_mean(encoded, observations[target], num_instances)
        log_counts = keras.ops.log1p(segment_count(observations[target], num_instances, pooled.dtype))

        result += [pooled, log_counts]

    for node in latent:
        if node in target_indices:
            result.append(gather(node_values(batch, node), target_indices[node]))

    return keras.ops.concatenate(result, axis=-1)


def split_values(samples, network: list[str], variable_dims: dict[str, dict[str, int]]) -> dict[str, dict]:
    values_by_node = {}
    start = 0

    for node in network:
        values = {}

        for variable in sorted(variable_dims[node]):
            dim = variable_dims[node][variable]
            values[variable] = samples[:, start : start + dim]
            start += dim

        values_by_node[node] = values

    return values_by_node


def network_variables(batch: dict, network: list[str]):
    values = []

    for node in network:
        values.append(node_values(batch, node))

    return keras.ops.concatenate(values, axis=-1)


def node_values(batch: dict, node: str):
    values = []

    for variable in sorted(batch[node]["values"]):
        values.append(batch[node]["values"][variable])

    return keras.ops.cast(keras.ops.concatenate(values, axis=-1), keras.config.floatx())
