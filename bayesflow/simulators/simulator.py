from collections.abc import Callable
import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import concatenate_dicts


class Simulator:
    def sample(self, batch_shape: Shape, *, numpy: bool = False, **kwargs) -> dict[str, Tensor]:
        raise NotImplementedError

    def rejection_sample(
        self,
        batch_shape: Shape,
        predicate: Callable[[dict[str, Tensor]], Tensor],
        *,
        axis: int = 0,
        numpy: bool = False,
        sample_size: int = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        if sample_size is None:
            sample_shape = batch_shape
        else:
            sample_shape = list(batch_shape)
            sample_shape[axis] = sample_size

            sample_shape = tuple(sample_shape)

        result = {}

        while not result or keras.ops.shape(next(iter(result.values())))[axis] < batch_shape[axis]:
            # get a batch of samples
            samples = self.sample(sample_shape, **kwargs)

            # get acceptance mask and turn into indices
            accept = predicate(samples)

            if not keras.ops.is_tensor(accept):
                raise RuntimeError("Predicate must return a tensor.")

            if not keras.ops.shape(accept) == (sample_shape[axis],):
                raise RuntimeError(
                    f"Predicate return tensor must have shape {(sample_shape[axis],)}. "
                    f"Received: {keras.ops.shape(accept)}."
                )

            if not keras.ops.dtype(accept) == "bool":
                # we could cast, but this tends to hide mistakes in the predicate
                raise RuntimeError("Predicate must return a tensor of dtype bool.")

            (accept,) = keras.ops.nonzero(accept)

            if not keras.ops.any(accept):
                # no samples accepted, skip
                continue

            # apply acceptance mask
            samples = {key: keras.ops.take(value, accept, axis=axis) for key, value in samples.items()}

            # concatenate with previous samples
            if not result:
                result = samples
            else:
                result = concatenate_dicts([result, samples], axis=axis)

        if numpy:
            result = {key: keras.ops.convert_to_numpy(value) for key, value in result.items()}

        return result
