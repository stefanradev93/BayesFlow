import keras
from bayesflow.experimental.types import Tensor
from bayesflow.experimental.utils import keras_kwargs
from keras import layers, Sequential, regularizers
from keras.saving import (register_keras_serializable)

@register_keras_serializable(package="bayesflow.networks") #TODO: finalize class name and add to package string
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
        num_filters: int,
        kernel_size: int = 4,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: regularizers.Regularizer | None = None,
        activation: str = "relu",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # TODO: Tidy code and condense comments
        
        # Define model sequencer
        model = Sequential()
                
        # 1D convolution layer with ReLU activation
        conv1 = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            activation="relu",
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )
        
        # Batch normalization layer
        bnorm = layers.BatchNormalization() # any custom ars here? 
        
        # GRU layers (create some sort of functional to add to Sequencer)
        full_gru = layers.GRU(...) # temp for gru.py
        skip_gru = layers.GRU(...)
        comb_gru = layers.Concatenate(axis=-1)
        
        # Final dense layer
        final_dense = layers.Dense(...)
        
        # Aggregate layers
        model.add(conv1)
        model.add(bnorm)
        
        # How to send to different channels for GRU(s)?
        
        model.add(final_dense)
        
        
        
        
        # Add in ResNet instead of final dense layer? (test for performance)
        
        
        pass
    
    def call(self, x: Tensor) -> Tensor:
        
        # Run through top layers
        # 2d convolution layer: (try 1d conv)
        
        # ReLU activation layer:
        
        # 2d batch norm layer:
        
        # Reshape
        
        # gru selection (calls gru and skip_gru classes)
        
        # combine results of above to dense layer
        
        pass
    
    
    def build(self, input_shape):
        # Built in reference to deepset changes
        super().build(input_shape)
        self(keras.KerasTensor(input_shape))
    
    
    # get_config???