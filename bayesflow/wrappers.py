# Copyright (c) 2022 The BayesFlow Developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf


class SpectralNormalization(tf.keras.layers.Wrapper):
    """Performs spectral normalization on neural network weights. Adapted from:

    https://www.tensorflow.org/addons/api_docs/python/tfa/layers/SpectralNormalization

    This wrapper controls the Lipschitz constant of a layer by
    constraining its spectral norm, which can stabilize the training of generative networks.

    See Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957).
    """

    def __init__(self, layer, power_iterations=1, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero, got " "`power_iterations={}`".format(power_iterations)
            )
        self.power_iterations = power_iterations
        self._initialized = False

    def build(self, input_shape):
        """Build `Layer`"""

        # Register input shape
        super().build(input_shape)

        # Store reference to weights
        if hasattr(self.layer, "kernel"):
            self.w = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.w = self.layer.embeddings
        else:
            raise AttributeError(
                "{} object has no attribute 'kernel' nor " "'embeddings'".format(type(self.layer).__name__)
            )

        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="sn_u",
            dtype=self.w.dtype,
        )

    def call(self, inputs, training=False):
        """Call `Layer`

        Parameters
        ----------

        inputs : tf.Tensor of shape (None,...,condition_dim + target_dim)
            The inputs to the corresponding layer.
        """

        if training:
            self.normalize_weights()
        output = self.layer(inputs)
        return output

    def normalize_weights(self):
        """Generate spectral normalized weights.

        This method will update the value of `self.w` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """

        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u

        with tf.name_scope("spectral_normalize"):
            for _ in range(self.power_iterations):
                v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
                u = tf.math.l2_normalize(tf.matmul(v, w))
            u = tf.stop_gradient(u)
            v = tf.stop_gradient(v)
            sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)
            self.u.assign(tf.cast(u, self.u.dtype))
            self.w.assign(tf.cast(tf.reshape(self.w / sigma, self.w_shape), self.w.dtype))

    def get_config(self):
        config = {"power_iterations": self.power_iterations}
        base_config = super().get_config()
        return {**base_config, **config}
