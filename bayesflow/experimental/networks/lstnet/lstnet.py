import keras
# from bayesflow.experimental.types import Tensor
from tensorflow import Tensor # TODO: remove, temp to stop strange Notebook errors
from bayesflow.experimental.utils import keras_kwargs
from keras import layers, Sequential, regularizers
from keras.saving import (register_keras_serializable)

@register_keras_serializable(package="bayesflow.networks.lstnet")
class LSTNet(keras.Model):
    """
    Implements a LSTNet Architecture as described in [1]
    
    [1] Y. Zhang and L. Mikelsons, Solving Stochastic Inverse Problems with Stochastic BayesFlow, 
    2023 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM), 
    Seattle, WA, USA, 2023, pp. 966-972, doi: 10.1109/AIM46323.2023.10196190.
    
    TODO: Add class spec and arg descriptions
    
    """
    
    # TODO: use new kwargs system
    
    def __init__(
        self,
        cnn_out: int, # C | R | O
        num_time_steps: int, # T
        kernel_size: int = 4, # F
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: regularizers.Regularizer | None = None,
        activation: str = "relu",
        gru_out: int = 64,
        dense_out: int = 64,
        skip_steps: list[int] = [1], # S
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # TODO: Tidy code and condense comments
        
        # Define model sequencer
        self.model = Sequential()
        
        # 1D convolution layer with custom activation
        self.conv1 = layers.Conv1D(
            filters=cnn_out,
            kernel_size=kernel_size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )
        
        # Batch normalization layer
        self.bnorm = layers.BatchNormalization() # TODO: any custom args here?
        
        # GRU layers
        self.keep_gru = layers.GRU(gru_out) # temp for gru.py
        self.skip_gru = layers.GRU(gru_out) # temp for skip_gru.py
        self.gru_add = layers.Add()
        # self.gru_concat = layers.Concatenate(axis=-1)
        
        # Final dense layer
        self.final_dense = layers.Dense(dense_out, activation="relu") # TODO: upgrade to ResNet
        
        # Aggregate layers
        #                                In:  (batch, time steps, num series)
        self.model.add(self.conv1)       # -> (batch, reduced time steps, cnn_out)
        self.model.add(self.bnorm)       # -> (batch, reduced time steps, cnn_out)
        self.model.add(self.keep_gru)    # -> (batch, gru_out)
        self.model.add(self.final_dense) # -> (batch, gru_out)
    
    
    def call(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x
    
    
    def build(self, input_shape):
        super().build(input_shape)
        self(keras.KerasTensor(input_shape))