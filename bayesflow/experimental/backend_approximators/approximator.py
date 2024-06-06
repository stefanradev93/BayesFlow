
import keras

match keras.backend.backend():
    case "jax":
        from .jax_approximator import JAXApproximator as Approximator
    case "numpy":
        from .numpy_approximator import NumpyApproximator as Approximator
    case "tensorflow":
        from .tensorflow_approximator import TensorFlowApproximator as Approximator
    case "torch":
        from .torch_approximator import TorchApproximator as Approximator
    case other:
        raise NotImplementedError(f"BayesFlow does not currently support backend '{other}'.")
