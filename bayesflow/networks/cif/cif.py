import keras
from keras.saving import register_keras_serializable
from ..inference_network import InferenceNetwork
from ..coupling_flow import CouplingFlow
from .conditional_gaussian import ConditionalGaussian


@register_keras_serializable(package="bayesflow.networks")
class CIF(InferenceNetwork):
    """Implements a continuously indexed flow (CIF) with a `CouplingFlow` bijection and
    `ConditionalGaussian` distributions p and q. Improves on eliminating leaky
    sampling found topologically in normalizing flows. Bulit in reference to [1].
    
    [1] R. Cornish, A. Caterini, G. Deligiannidis, & A. Doucet (2021).
    Relaxing Bijectivity Constraints with Continuously Indexed Normalising Flows.
    arXiv:1909.13833.
    """
    
    def __init__(self, pq_depth=4, pq_width=128, pq_activation="tanh", **kwargs):
        """Creates an instance of a `CIF` with configurable `ConditionalGaussian` distributions
        p and q, each containing MLP networks

        Parameters:
        -----------
        pq_depth: int, optional, default: 4
            The number of MLP hidden layers (minimum: 1)
        pq_width: int, optional, default: 128
            The dimensionality of the MLP hidden layers
        pq_activation: str, optional, default: 'tanh'
            The MLP activation function
        """
        
        super().__init__(base_distribution="normal", **kwargs)
        self.bijection = CouplingFlow()
        self.p_dist = ConditionalGaussian(depth=pq_depth, width=pq_width, activation=pq_activation)
        self.q_dist = ConditionalGaussian(depth=pq_depth, width=pq_width, activation=pq_activation)
        
    
    def build(self, xz_shape, conditions_shape=None):
        super().build(xz_shape)        
        self.bijection.build(xz_shape, conditions_shape=conditions_shape)
        self.p_dist.build(xz_shape)
        self.q_dist.build(xz_shape)
        
    
    def call(self, xz, conditions=None, inverse=False, **kwargs):
        if inverse:
            return self._inverse(xz, conditions=conditions, **kwargs)
        return self._forward(xz, conditions=conditions, **kwargs)
    
    
    def _forward(self, x, conditions=None, density=False, **kwargs):
        # Sample u ~ q_u
        u, log_qu = self.q_dist.sample(x, log_prob=True)
                
        # Bijection and log jacobian x -> z
        z, log_jac = self.bijection(x, conditions=conditions, density=True)
        if log_jac.ndim > 1:
            log_jac = keras.ops.sum(log_jac, axis=1)
        
        # Log prob over p on u with conditions z
        log_pu = self.p_dist.log_prob(u, z)
        
        # Prior log prob
        log_prior = self.base_distribution.log_prob(z)
        if log_prior.ndim > 1:
            log_prior = keras.ops.sum(log_prior, axis=1)
        
        # ELBO loss
        elbo = log_jac + log_pu + log_prior - log_qu
        
        if density:
            return z, elbo
        return z
    
    
    def _inverse(self, z, conditions=None, density=False, **kwargs):
        # Inverse bijection z -> x
        u = self.p_dist.sample(z)
        x = self.bijection(z, conditions=conditions, inverse=True)
        if density:
            log_pu = self.p_dist.log_prob(u, x)
            return x, log_pu
        return x
    
    
    def compute_metrics(self, data, stage="training"):
        base_metrics = super().compute_metrics(data, stage=stage)
        inference_variables = data["inference_variables"]
        inference_conditions = data.get("inference_conditions")
        _, elbo = self(inference_variables, conditions=inference_conditions, inverse=False, density=True)
        loss = -keras.ops.mean(elbo)
        return base_metrics | {"loss": loss}
