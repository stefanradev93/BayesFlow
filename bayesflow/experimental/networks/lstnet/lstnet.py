import keras
# from bayesflow.experimental.types import Tensor
from tensorflow import Tensor # TODO: remove, temp to stop strange Notebook errors
from bayesflow.experimental.utils import keras_kwargs
from keras import layers, Sequential, regularizers
from keras.saving import (register_keras_serializable)
from .skip_gru import SkipGRU
from ...networks.resnet import ResNet

@register_keras_serializable(package="bayesflow.networks.lstnet")
class LSTNet(keras.Model):
    """
    Implements a LSTNet Architecture as described in [1]
    
    [1] Y. Zhang and L. Mikelsons, Solving Stochastic Inverse Problems with Stochastic BayesFlow, 
    2023 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM), 
    Seattle, WA, USA, 2023, pp. 966-972, doi: 10.1109/AIM46323.2023.10196190.
    
    TODO: Add proper docstring
    
    """
    
    # TODO: use new kwargs system
    
    def __init__(
        self,
        cnn_out: int, # C | R | O
        kernel_size: int = 4, # F
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: regularizers.Regularizer | None = None,
        activation: str = "relu",
        gru_out: int = 64,
        resnet_out: int = 32,
        skip_steps: list[int] = [2], # S
        **kwargs
    ):
        super().__init__(**kwargs)
                
        # Define model
        self.model = Sequential()
        self.conv1 = layers.Conv1D(
            filters=cnn_out,
            kernel_size=kernel_size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )        
        self.bnorm = layers.BatchNormalization()        
        self.skip_gru = SkipGRU(gru_out, skip_steps)        
        self.resnet = ResNet(width=resnet_out)
        
        # Aggregate layers               In:  (batch, time steps, num series)
        self.model.add(self.conv1)       # -> (batch, reduced time steps, cnn_out)
        self.model.add(self.bnorm)       # -> (batch, reduced time steps, cnn_out)
        self.model.add(self.skip_gru)    # -> (batch, gru_out)
        self.model.add(self.resnet)      # -> (batch, resnet_out)
    
    def call(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x
    
    def build(self, input_shape):
        self.call(keras.ops.zeros(input_shape))