import keras
import numpy as np


class LogGradientNorm(keras.callbacks.Callback):
    def __init__(self, ord: int = 2):
        super().__init__()
        self.ord = ord

    def on_train_batch_end(self, batch, logs=None):
        match keras.backend.backend():
            case "torch":
                gradients = [weight.value.grad for weight in self.model.weights]
                gradients = [gradient.flatten().numpy() for gradient in gradients]

                gradients = np.concatenate(gradients)

                gradient_norm = np.linalg.norm(gradients, ord=self.ord)
            case other:
                raise NotImplementedError(f"Gradient norm logging for backend '{other}' is not yet supported.")

        logs["gradient_norm"] = gradient_norm
