import keras

from bayesflow.utils.serialization import serializable


def gather(values, rows):
    return keras.ops.take(values, rows, axis=0)


def segment_count(ids, num_segments, dtype):
    ones = keras.ops.ones((keras.ops.shape(ids)[0], 1), dtype=dtype)

    return keras.ops.segment_sum(ones, ids, num_segments=num_segments)


def segment_mean(values, ids, num_segments):
    totals = keras.ops.segment_sum(values, ids, num_segments=num_segments)
    counts = segment_count(ids, num_segments, totals.dtype)

    return totals / keras.ops.maximum(counts, 1.0)


def sum_per_dataset(values, datasets, num_datasets):
    values = keras.ops.reshape(values, (keras.ops.shape(values)[0], -1))

    return keras.ops.segment_sum(keras.ops.sum(values, axis=-1), datasets, num_segments=num_datasets)


@serializable("bayesflow.experimental")
class ExchangeableEncoder(keras.Layer):
    def __init__(self, width: int = 64, depth: int = 2, activation: str = "mish", **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.activation = activation

    def build(self, observations_shape, groups_shape):
        num_groupings = len(groups_shape)
        dim = observations_shape[-1] + num_groupings

        self.kernels = []
        self.group_kernels = []
        self.biases = []

        for i in range(self.depth):
            self.kernels.append(self.add_weight(shape=(dim, self.width), name=f"kernel_{i}"))
            self.group_kernels.append(self.add_weight(shape=(num_groupings, dim, self.width), name=f"group_kernel_{i}"))
            self.biases.append(self.add_weight(shape=(self.width,), initializer="zeros", name=f"bias_{i}"))
            dim = self.width

    def call(self, observations, groups, training=False):
        activation = keras.activations.get(self.activation)

        features = [observations]

        for group in groups:
            log_size = keras.ops.log1p(keras.ops.cast(group["counts"], observations.dtype))
            features.append(gather(log_size, group["ids"])[:, None])

        hidden = keras.ops.concatenate(features, axis=-1)

        # every layer adds a transformation of the group means of each grouping to every row
        for kernel, group_kernel, bias in zip(self.kernels, self.group_kernels, self.biases):
            pooled = bias

            for i, group in enumerate(groups):
                num_groups = keras.ops.shape(group["counts"])[0]
                mean = segment_mean(hidden, group["ids"], num_groups)
                pooled = pooled + gather(keras.ops.matmul(mean, group_kernel[i]), group["ids"])

            hidden = activation(keras.ops.matmul(hidden, kernel) + pooled)

        return hidden

    def get_config(self):
        return super().get_config() | {"width": self.width, "depth": self.depth, "activation": self.activation}
