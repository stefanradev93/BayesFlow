import keras

from .encoder import gather, segment_count, segment_mean


def network_conditions(
    batch: dict, network: list[str], conditions: list[str], data_node: str, encoder: keras.Layer, training=False
):
    observations = batch[data_node]["indices"]
    target = network[0]
    target_indices = batch[target]["indices"]

    latent = []

    for node in conditions:
        if node != data_node:
            latent.append(node)

    # extend observations with conditions
    features = [node_values(batch, data_node)]

    for node in latent:
        features.append(gather(node_values(batch, node), observations[node]))

    # encode extended observations
    groups = []

    for name in sorted(batch[data_node]["groups"]):
        groups.append(batch[data_node]["groups"][name])

    encoded = encoder(keras.ops.concatenate(features, axis=-1), groups, training=training)

    # average and append log number
    num_instances = keras.ops.shape(target_indices[target])[0]
    pooled = segment_mean(encoded, observations[target], num_instances)
    log_counts = keras.ops.log1p(segment_count(observations[target], num_instances, pooled.dtype))

    result = [pooled, log_counts]

    # add unpooled conditions
    for node in latent:
        if node in target_indices:
            result.append(gather(node_values(batch, node), target_indices[node]))

    return keras.ops.concatenate(result, axis=-1)


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
