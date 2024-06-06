import keras
from bayesflow.experimental.types import Tensor
from bayesflow.experimental.utils import keras_kwargs
from keras import layers, Sequential
from keras.saving import (register_keras_serializable)

@register_keras_serializable(package="bayesflow.networks") #TODO: finalize class name and add to package string
class LSTNet(keras.Model):
    """
    Implements a LSTNet Architecture as described in [1]
    
    [1] Y. Zhang and L. Mikelsons, Solving Stochastic Inverse Problems with Stochastic BayesFlow, 
    2023 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM), 
    Seattle, WA, USA, 2023, pp. 966-972, doi: 10.1109/AIM46323.2023.10196190.
    
    """
    
    # TODO: use new kwargs system
    
    def __init__(
        self,
        kernel_size: int = 4,
        activation: str = "relu",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Define layers here
        model = Sequential()
        
        # Convolution (with relu) and dense layers (for debugging)
        
        # 1D convolution layer with ReLU activation
        conv1 = layers.Conv1D(activation="relu")
        
        # Batch normalization layer
        bnorm = layers.BatchNormalization(...)
        
        # GRU layers
        full_gru = layers.GRU(...)
        skip_gru = layers.GRU(...)
        
        # Final dense layer
        final_dense = layers.Dense(...)
        
        # Aggregate layers
        model.add(conv1)
        model.add(bnorm)
        
        # How to send to different channels?
        
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