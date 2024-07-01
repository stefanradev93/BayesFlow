import keras
import numpy as np
from bayesflow.types import Tensor

from .integrator import Integrator
class RKIntegrator(Integrator):
    def __init__(self, order: int = 4):
        self.order = order
        # TODO: order must either 3 or 4
    
    def integrate(self, x: Tensor, steps: int, conditions: Tensor = None, dynamic: bool = False):
        raise NotImplementedError("TODO")