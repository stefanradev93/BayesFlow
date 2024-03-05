
import keras

backend = keras.backend.backend()

match backend:
    case "jax":
        from .jax import JAXGenerativeTrainer as GenerativeTrainer
    case "tensorflow":
        from .tensorflow import TensorFlowGenerativeTrainer as GenerativeTrainer
    case "torch":
        from .pytorch import PyTorchGenerativeTrainer as GenerativeTrainer
    case other:
        raise NotImplementedError(f"Backend '{backend}' must implement a GenerativeTrainer.")
