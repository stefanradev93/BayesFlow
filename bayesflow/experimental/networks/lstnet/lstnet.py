import keras
# from bayesflow.experimental.types import Tensor
from tensorflow import Tensor
from bayesflow.experimental.utils import keras_kwargs
from keras import layers, Sequential, regularizers
from keras.saving import (register_keras_serializable)

@register_keras_serializable(package="bayesflow.networks.lstnet") #TODO: finalize class name and add to package string
class LSTNet(keras.Model):
    """
    Implements a LSTNet Architecture as described in [1]
    
    [1] Y. Zhang and L. Mikelsons, Solving Stochastic Inverse Problems with Stochastic BayesFlow, 
    2023 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM), 
    Seattle, WA, USA, 2023, pp. 966-972, doi: 10.1109/AIM46323.2023.10196190.
    
    TODO: Add class spec and arg descriptions
    
    """
    
    # TODO: use new kwargs system
    # TODO: currently assumes cnn output dim will match gru output dim. Do these have to be different?
    # TODO: maybe change conv_out to hidden_dim, and add a separate out_dim required variable
    
    def __init__(
        self,
        cnn_out: int, # C | R | O
        num_time_steps: int, # T
        kernel_size: int = 4, # F
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: regularizers.Regularizer | None = None,
        activation: str = "relu",
        conv_out: int = 64,
        skip_step: int = 6, # S
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
        
        # Reshaping layers
        # TODO: skeptical on these dimensions being correct
        self.keep_reshape = layers.Reshape((-1, num_time_steps - kernel_size + 1, conv_out))
        self.skip_reshape = layers.Reshape((-1, (num_time_steps - kernel_size + 1) // skip_step, conv_out * skip_step))
        
        # GRU layers
        self.keep_gru = layers.GRU(conv_out) # temp for gru.py
        self.skip_gru = layers.GRU(conv_out) # temp for skip_gru.py
        self.gru_add = layers.Add()
        # self.gru_concat = layers.Concatenate(axis=-1)
        
        # Final dense layer
        self.final_dense = layers.Dense(cnn_out, activation="relu") # TODO: upgrade to ResNet
        
        # Aggregate layers
        # self.model.add(conv1)
        # self.model.add(bnorm)
        
        # How to send to different channels for GRU(s)?
    
    def call(self, x: Tensor) -> Tensor:
        print("Performing pass with shape: ", x.shape)
        
        # Conv and batch norm
        # Input: (2, 32, 2)
        x1 = self.conv1(x) # (2, 29, 32)
        x2 = self.bnorm(x1) # (2, 29, 32)
        # x = self.model(x)
        # Reshape into parallel channels
        x_keep = self.keep_reshape(x2)
        x_skip = self.skip_reshape(x2)
        # Parallel GRU and Skip GRU
        x_keep = self.keep_gru(x_keep)
        x_skip = self.skip_gru(x_skip)
        # Concatenate results
        x3 = self.gru_add([x_keep, x_skip])
        # Final dense and return
        x4 = self.final_dense(x3)
        return x4
    
    
    def build(self, input_shape):
        # Built in reference to deepset changes
        super().build(input_shape)
        self(keras.KerasTensor(input_shape))