import keras

match keras.backend.backend():
    case "jax":
        from .jax_approximator import JAXApproximator as BackendWorkflow
    case "numpy":
        from .numpy_approximator import NumpyApproximator as BackendWorkflow
    case "tensorflow":
        from .tensorflow_approximator import TensorFlowApproximator as BackendWorkflow
    case "torch":
        from .torch_approximator import TorchApproximator as BackendWorkflow
    case other:
        raise ValueError(f"Backend '{other}' does not support workflows.")
