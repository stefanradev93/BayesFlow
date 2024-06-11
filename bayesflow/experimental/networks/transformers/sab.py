
from keras.saving import (
    deserialize_keras_object,
    register_keras_serializable,
    serialize_keras_object,
)

from .mab import MultiHeadAttentionBlock

@register_keras_serializable(package="bayesflow.networks")
class SetAttentionBlock(MultiHeadAttentionBlock):
    """Implements the SAB block from [1] which represents learnable self-attention.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def call(self, set_x, **kwargs):
        """Performs the forward pass through the self-attention layer.

        Parameters
        ----------
        set_x   : Tensor
            Input of shape (batch_size, set_size, input_dim)

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, set_size, input_dim)
        """

        return super().call(set_x, set_x, **kwargs)