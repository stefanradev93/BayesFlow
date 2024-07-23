import keras
from ..inference_network import InferenceNetwork
from ..coupling_flow import CouplingFlow


class CIF(InferenceNetwork):
    def __init__(self, **kwargs):
        super().__init__(base_distribution="normal", **kwargs)
        # Member variables wrt to nux implementation
        self.feature_net = CouplingFlow() 	 # no conditions
        self.flow = CouplingFlow() 			 # bijective transformer
        self.u_dist = self.base_distribution # Gaussian prior
        self.v_dist = CouplingFlow()		 # conditioned flow / parameterized gaussian
        
    
    def build(self, xz_shape, conditions_shape):
        super().build(xz_shape)            
        self.feature_net.build(xz_shape)
        self.flow.build(xz_shape, xz_shape)
        self.v_dist.build(xz_shape, xz_shape)
        
    
    def call(self, xz, conditions, inverse=False, **kwargs):
        if inverse:
            return self._inverse(xz, conditions, **kwargs)
        return self._forward(xz, conditions, **kwargs)
    
    
    def _forward(self, x, conditions, density=False, **kwargs):
        # NOTE: conditions should be used...
        
        # Sample u ~ q(u|phi_x)
        phi_x = self.feature_net(x, conditions=None)
        u, log_qu = self.v_dist(keras.ops.zeros_like(x), conditions=phi_x, inverse=True, density=True)
        
        # Compute z = f(x; phi_u) and p(x|u)
        phi_u = self.feature_net(u, conditions=None)
        z, log_px = self.flow(x, conditions=phi_u, inverse=False, density=True)
        
        # Compute p(u)
        log_pu = self.base_distribution.log_prob(u)
        
        # Log likelihood?
        llc = log_px + log_pu - log_qu
        
        # NOTE - this can be moved up when I'm done tinkering
        if density:
            return z, llc
        return z
    
    
    def _inverse(self, z, conditions, density=False, **kwargs):
        # NOTE: conditions should be used...
        
        # Sample u ~ p(u)
        u = self.base_distribution.sample(keras.ops.shape(z))
        log_pu = self.base_distribution.log_prob(keras.ops.zeros_like(z))
        
        # Compute inverse of f(z; u)
        phi_u = self.feature_net(u)
        x, log_px = self.flow(z, conditions=phi_u, inverse=True, density=True)
        
        # Predict q(u|x)
        phi_x = self.feature_net(x)
        _, log_qu = self.v_dist(u, conditions=phi_x, inverse=False, density=True)
        
        # Log likelihood?
        llc = log_px + log_pu - log_qu
        
        # NOTE: this can be moved up when I'm done tinkering
        if density:
            return x, llc
        return x
    
    
    def compute_metrics(self, data, stage="training"):
        base_metrics = super().compute_metrics(data, stage=stage)
        inference_variables = data["inference_variables"]
        inference_conditions = data.get("inference_conditions")
        
        z, log_density = self(inference_variables, conditions=inference_conditions, inverse=False, density=True)
        # Should loss be reduced this way..?
        loss = -keras.ops.mean(log_density)
        return base_metrics | {"loss": loss}
        
        