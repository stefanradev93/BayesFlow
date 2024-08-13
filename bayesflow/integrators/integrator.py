import keras
import numpy as np
from bayesflow.types import Tensor, Shape
from bayesflow.utils import keras_kwargs

class Integrator(keras.Layer):
    def __init__(self, **kwargs):
        super().__init__(**keras_kwargs(kwargs))
    
    def call(self, x: Tensor, steps: int, conditions: Tensor = None, dynamic: bool = False):
        raise NotImplementedError
