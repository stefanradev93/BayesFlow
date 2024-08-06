import keras
from keras.saving import register_keras_serializable
import numpy as np
from ..mlp import MLP
from bayesflow.utils import keras_kwargs


@register_keras_serializable(package="bayesflow.networks.cif")
class ConditionalGaussian(keras.Layer):
    """Implements a conditional gaussian distribution with neural networks for the
    means and standard deviations respectively. Bulit in reference to [1].
    
    [1] R. Cornish, A. Caterini, G. Deligiannidis, & A. Doucet (2021).
    Relaxing Bijectivity Constraints with Continuously Indexed Normalising Flows.
    arXiv:1909.13833.
    """
    
    def __init__(self, depth=4, width=128, activation="tanh", **kwargs):
        """Creates an instance of a `ConditionalGaussian` with configurable `MLP`
        networks for the means and standard deviations.

        Parameters:
        -----------
        depth: int, optional, default: 4
            The number of MLP hidden layers (minimum: 1)
        width: int, optional, default: 128
            The dimensionality of the MLP hidden layers
        activation: str, optional, default: "tanh"
            The MLP activation function
        """
        
        super().__init__(**keras_kwargs(kwargs))
        self.means = MLP(depth=depth, width=width, activation=activation)
        self.stds = MLP(depth=depth, width=width, activation=activation)
        self.output_projector = keras.layers.Dense(None)
    
    
    def build(self, input_shape):
        self.means.build(input_shape)
        self.stds.build(input_shape)
        self.output_projector.units = input_shape[-1]
    
    
    def _diagonal_gaussian_log_prob(self, conditions, means, stds):
        flat_c = keras.layers.Flatten()(conditions)
        flat_means = keras.layers.Flatten()(means)
        flat_vars = keras.layers.Flatten()(stds) ** 2
        
        dim = keras.ops.shape(flat_c)[1]
        
        const_term = -0.5 * dim * np.log(2 * np.pi)
        log_det_terms = -0.5 * keras.ops.sum(keras.ops.log(flat_vars), axis=1)
        product_terms = -0.5 * keras.ops.sum((flat_c - flat_means) ** 2 / flat_vars, axis=1)
        
        return const_term + log_det_terms + product_terms


    def log_prob(self, x, conditions):
        means = self.output_projector(self.means(conditions))
        stds = keras.ops.exp(self.output_projector(self.stds(conditions)))
        return self._diagonal_gaussian_log_prob(x, means, stds)

    
    def sample(self, conditions, log_prob=False):
        means = self.output_projector(self.means(conditions))
        stds = keras.ops.exp(self.output_projector(self.stds(conditions)))
        
        # Reparameterize
        samples = stds * keras.random.normal(keras.ops.shape(conditions)) + means
        
        # Log probability
        if log_prob:
            log_p = self._diagonal_gaussian_log_prob(samples, means, stds)
            return samples, log_p
        
        return samples