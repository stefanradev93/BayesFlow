import keras

from bayesflow.utils import logging


match keras.backend.backend():
    case "jax":
        from .jax_approximator import JAXApproximator as BaseBackendApproximator
    case "numpy":
        from .numpy_approximator import NumpyApproximator as BaseBackendApproximator
    case "tensorflow":
        from .tensorflow_approximator import TensorFlowApproximator as BaseBackendApproximator
    case "torch":
        from .torch_approximator import TorchApproximator as BaseBackendApproximator
    case other:
        raise ValueError(f"Backend '{other}' does not support workflows.")


class BackendApproximator(BaseBackendApproximator):
    def fit(self, *, dataset: keras.utils.PyDataset, **kwargs):
        if not self.built:
            logging.info("Automatically building networks based on a test batch.")
            test_batch = dataset[0]
            self.build({key: keras.ops.shape(value) for key, value in test_batch.items()})

        return super().fit(x=dataset, y=None, **kwargs)
