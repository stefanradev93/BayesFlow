import keras
import numpy as np
from bayesflow.types import Tensor

class Integrator():
    def integrate(self, x: Tensor, steps: int, conditions: Tensor = None, dynamic: bool = False):
        raise NotImplementedError
