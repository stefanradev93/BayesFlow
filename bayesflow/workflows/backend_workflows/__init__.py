import keras

match keras.backend.backend():
    case "jax":
        from .jax_workflow import JAXWorkflow as BackendWorkflow
    case "numpy":
        from .numpy_workflow import NumpyWorkflow as BackendWorkflow
    case "tensorflow":
        from .tensorflow_workflow import TensorFlowWorkflow as BackendWorkflow
    case "torch":
        from .torch_workflow import TorchWorkflow as BackendWorkflow
    case other:
        raise ValueError(f"Backend '{other}' does not support workflows.")
