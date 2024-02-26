
import keras

from .generative_model import GenerativeModel


class SimulationBasedInference(keras.Model):
    def __init__(self, generative_model: GenerativeModel) -> None:
        super().__init__()
        self.generative_model = generative_model

