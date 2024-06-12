import keras
from keras.saving import register_keras_serializable
from keras import layers, Sequential
from bayesflow.experimental.types import Tensor
from bayesflow.experimental.utils import keras_kwargs

@register_keras_serializable(package="bayesflow.networks.skip_gru")
class SkipGRU(keras.Model):
    def __init__(self, gru_out: int, skip_steps: list[int], **kwargs):
        super().__init__(**keras_kwargs(kwargs))
        self.gru_out = gru_out
        self.skip_steps = skip_steps
        self.gru = layers.GRU(gru_out)
        self.skip_grus = [layers.GRU(gru_out) for _ in range(len(self.skip_steps))]
        
    def call(self, x: Tensor) -> Tensor:
        # Standard GRU
        # In: (batch, reduced time steps, cnn_out)
        gru = self.gru(x) # -> (batch, gru_out)
                
        # Skip GRU
        for i, skip_step in enumerate(self.skip_steps):
            # Reshape, remove skipped time points
            skip_length = x.shape[1] // skip_step
            s = x[:, -skip_length * skip_step:, :] # -> (batch, shrinked time steps, cnn_out)
            s1 = keras.ops.reshape(s, (-1, s.shape[2], skip_length, skip_step)) # -> (batch, cnn_out, skip_length, skip_step)
            s2 = keras.ops.transpose(s1, [0, 3, 2, 1]) # -> (batch, skip step, skip_length, cnn_out)
            s3 = keras.ops.reshape(s2, (-1, s2.shape[2], s2.shape[3])) # -> (batch * skip step, skip_length, cnn_out)
            
            # GRU on remaining data
            s4 = self.skip_grus[i](s3) # -> (batch * skip step, gru_out)
            s5 = keras.ops.reshape(s4, (-1, skip_step * s4.shape[1])) # -> (batch, skip step * gru_out)
            
            # Concat
            gru = keras.ops.concatenate([gru, s5], axis=1) # -> (batch, gru_out * skip step * 2)
            
        return gru    
    
    def build(self, input_shape):
        self.call(keras.ops.zeros(input_shape))