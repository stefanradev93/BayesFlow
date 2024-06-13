import keras
from keras.saving import register_keras_serializable
from keras import layers
from bayesflow.experimental.types import Tensor
from bayesflow.experimental.utils import keras_kwargs

@register_keras_serializable(package="bayesflow.networks.skip_gru")
class SkipGRU(keras.Model):
    """
    Implements a Skip GRU layer as described in [1]
    
    [1] Y. Zhang and L. Mikelsons, Solving Stochastic Inverse Problems with Stochastic BayesFlow, 
    2023 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM), 
    Seattle, WA, USA, 2023, pp. 966-972, doi: 10.1109/AIM46323.2023.10196190.
    
    TODO: Add proper docstring
    
    """
    def __init__(self, gru_out: int, skip_outs: list[int], skip_steps: list[int], **kwargs):        
        super().__init__(**keras_kwargs(kwargs))
        self.gru_out = gru_out
        self.skip_steps = skip_steps
        self.gru = layers.GRU(gru_out)
        self.skip_grus = [layers.GRU(skip_outs[i]) for i in range(len(self.skip_steps))]
        
    def call(self, x: Tensor) -> Tensor:
        sgru = self.gru(x)
        for i, skip_step in enumerate(self.skip_steps):
            # Reshape, remove skipped time points
            skip_length = x.shape[1] // skip_step
            s = x[:, -skip_length * skip_step:, :]
            s = keras.ops.reshape(s, (-1, s.shape[2], skip_length, skip_step))
            s = keras.ops.transpose(s, [0, 3, 2, 1])
            s = keras.ops.reshape(s, (-1, s.shape[2], s.shape[3]))
            
            # Reapply GRU, add to working tensor
            s = self.skip_grus[i](s)
            s = keras.ops.reshape(s, (-1, skip_step * s.shape[1]))
            sgru = keras.ops.concatenate([sgru, s], axis=1)
            
        return sgru
    
    def build(self, input_shape):
        self.call(keras.ops.zeros(input_shape))