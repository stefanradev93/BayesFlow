import keras

match keras.backend.backend():
    case "jax":
        from .jax_approximator import JAXApproximator as BackendApproximator
    case "numpy":
        from .numpy_approximator import NumpyApproximator as BackendApproximator
    case "tensorflow":
        from .tensorflow_approximator import TensorFlowApproximator as BackendApproximator
    case "torch":
        from .torch_approximator import TorchApproximator as BackendApproximator
    case other:
        raise ValueError(f"Backend '{other}' does not support workflows.")
