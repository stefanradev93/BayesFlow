import keras
from keras import Sequential, Model
from keras.api.layers import Dense
from bayesflow.exceptions import ConfigurationError
from functools import partial
import tensorflow as tf

class InvariantModule(keras.Model):
    """Implements an invariant module performing a permutation-invariant transform.

    For details and rationale, see:

    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(self, settings, **kwargs):
        """Creates an invariant module according to [1] which represents a learnable permutation-invariant
        function with an option for learnable pooling.

        Parameters
        ----------
        settings : dict
            A dictionary holding the configuration settings for the module.
        **kwargs : dict, optional, default: {}
            Optional keyword arguments passed to the `tf.keras.Model` constructor.
        """

        super().__init__(**kwargs)

        # Create internal functions
        self.s1 = Sequential([Dense(**settings["dense_s1_args"]) for _ in range(settings["num_dense_s1"])])
        self.s2 = Sequential([Dense(**settings["dense_s2_args"]) for _ in range(settings["num_dense_s2"])])

        # Pick pooling function
        if settings["pooling_fun"] == "mean":
            pooling_fun = partial(tf.reduce_mean, axis=-2)
        elif settings["pooling_fun"] == "max":
            pooling_fun = partial(tf.reduce_max, axis=-2)
        else:
            if callable(settings["pooling_fun"]):
                pooling_fun = settings["pooling_fun"]
            else:
                raise ConfigurationError("pooling_fun argument not understood!")
        self.pooler = pooling_fun

    def call(self, x, **kwargs):
        """Performs the forward pass of a learnable invariant transform.

        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size,..., x_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size,..., out_dim)
        """

        x_reduced = self.pooler(self.s1(x, **kwargs))
        out = self.s2(x_reduced, **kwargs)
        return out