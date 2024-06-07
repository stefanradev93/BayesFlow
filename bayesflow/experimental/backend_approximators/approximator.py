
import keras

from bayesflow.experimental.configurators import Configurator
from bayesflow.experimental.types import Tensor

match keras.backend.backend():
    case "jax":
        from .jax_approximator import JAXApproximator as BaseApproximator
    case "numpy":
        from .numpy_approximator import NumpyApproximator as BaseApproximator
    case "tensorflow":
        from .tensorflow_approximator import TensorFlowApproximator as BaseApproximator
    case "torch":
        from .torch_approximator import TorchApproximator as BaseApproximator
    case other:
        raise NotImplementedError(f"BayesFlow does not currently support backend '{other}'.")


class Approximator(BaseApproximator):
    def __init__(self, inference_variables: list[str], inference_conditions: list[str] = None, summary_variables: list[str] = None, summary_conditions: list[str] = None, **kwargs):
        configurator = Configurator(inference_variables, inference_conditions, summary_variables, summary_conditions)
        kwargs.setdefault("summary_network", None)
        super().__init__(configurator=configurator, **kwargs)
