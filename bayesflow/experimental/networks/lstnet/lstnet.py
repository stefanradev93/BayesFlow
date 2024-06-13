import keras
from bayesflow.experimental.types import Tensor
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
        
    def __init__(
        self,
        cnn_out: int = 128,
        kernel_size: int = 4,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: regularizers.Regularizer | None = None,
        activation: str = "relu",
        gru_out: int = 64,
        skip_outs: list[int] = [32],
        skip_steps: list[int] = [2],
        resnet_out: int = 32,
        **kwargs
    ):
        if len(skip_outs) != len(skip_steps):
            raise ValueError("hidden_out must have same length as skip_steps")
        
        super().__init__(**keras_kwargs(kwargs))
                
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
        self.skip_gru = SkipGRU(gru_out, skip_outs, skip_steps)        
        self.resnet = ResNet(width=resnet_out)
        
        # Aggregate layers               In:  (batch, time steps, num series)
        self.model.add(self.conv1)       # -> (batch, reduced time steps, cnn_out)
        self.model.add(self.bnorm)       # -> (batch, reduced time steps, cnn_out)
        self.model.add(self.skip_gru)    # -> (batch, _)
        self.model.add(self.resnet)      # -> (batch, resnet_out)
    
    def call(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x
    
    def build(self, input_shape):
        self.call(keras.ops.zeros(input_shape))