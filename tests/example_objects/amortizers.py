import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from bayesflow.networks import Permutation


class InvariantCouplingNet(tf.keras.Model):
    """Implements a conditional version of a sequential network."""

    def __init__(self, meta, n_out):
        """
        Creates a conditional coupling net (FC neural network).
        ----------

        Arguments:
        meta  : list -- a list of dictionaries, wherein each dictionary holds parameter - value pairs for a single
                       tf.keras.Dense layer.
        n_out : int  -- number of outputs of the coupling net
        """

        super(InvariantCouplingNet, self).__init__()

        self.h1 = Sequential([Dense(**meta['dense_h1_args']) for _ in range(meta['n_dense_h1'])])
        self.h2 = Sequential(
            [Dense(**meta['dense_h2_args']) for _ in range(meta['n_dense_h2'])] +
            [Dense(n_out)]
        )

    def call(self, m, params, x):
        """
        Concatenates x and y and performs a forward pass through the coupling net.
        Arguments:
        m      : tf.Tensor of shape (batch_size, n_models) -- the one-hot-encoded model indices
        params : tf.Tensor of shape (batch_size, theta_dim)  -- the parameters theta ~ p(theta) of interest
        x      : tf.Tensor of shape (batch_size, n_obs, inp_dim) -- the conditional data of interest x
        """

        N = int(x.shape[1])
        params_rep = tf.stack([params] * N, axis=1)
        m_rep = tf.stack([m] * N, axis=1)
        x_params_m = tf.concat([x, params_rep, m_rep], axis=-1)
        rep = tf.reduce_mean(self.h1(x_params_m), axis=1)
        rep_params_m = tf.concat([rep, params, m], axis=-1)
        out = self.h2(rep_params_m)
        return out


class ConditionalCouplingLayer(tf.keras.Model):
    """Implements a conditional version of the INN block."""

    def __init__(self, meta):
        """
        Creates a conditional invertible block.
        ----------

        Arguments:
        meta      : list -- a list of dictionaries, wherein each dictionary holds parameter - value pairs for a single
                       tf.keras.Dense layer. All coupling nets are assumed to be equal.
        """

        super(ConditionalCouplingLayer, self).__init__()
        self.alpha = meta['alpha']
        theta_dim = meta['n_params']
        self.n_out1 = theta_dim // 2
        self.n_out2 = theta_dim // 2 if theta_dim % 2 == 0 else theta_dim // 2 + 1
        if meta['permute']:
            self.permutation = Permutation(theta_dim)
        else:
            self.permutation = None

        self.s1 = InvariantCouplingNet(meta['s_args'], self.n_out1)
        self.t1 = InvariantCouplingNet(meta['t_args'], self.n_out1)
        self.s2 = InvariantCouplingNet(meta['s_args'], self.n_out2)
        self.t2 = InvariantCouplingNet(meta['t_args'], self.n_out2)

    def call(self, m, params, x, inverse=False, log_det_J=True):
        """
        Implements both directions of a conditional invertible block.
        ----------

        Arguments:
        m         : tf.Tensor of shape (batch_size, n_models) -- the one-hot-encoded model indices
        theta     : tf.Tensor of shape (batch_size, theta_dim) -- the parameters theta ~ p(theta|y) of interest
        x         : tf.Tensor of shape (batch_size, summary_dim) -- the summarized conditional data of interest x = sum(x)
        inverse   : bool -- flag indicating whether to tun the block forward or backwards
        log_det_J : bool -- flag indicating whether to return the log determinant of the Jacobian matrix
        ----------

        Returns:
        (v, log_det_J)  :  (tf.Tensor of shape (batch_size, inp_dim), tf.Tensor of shape (batch_size, )) --
                           the transformed input, if inverse = False, and the corresponding Jacobian of the transformation
                            if inverse = False
        u               :  tf.Tensor of shape (batch_size, inp_dim) -- the transformed out, if inverse = True
        """

        # --- Forward pass --- #
        if not inverse:

            if self.permutation is not None:
                params = self.permutation(params)

            u1, u2 = tf.split(params, [self.n_out1, self.n_out2], axis=-1)

            # Pre-compute network outputs for v1
            s1 = self.s1(m, u2, x)
            # Clamp s1 if specified
            if self.alpha is not None:
                s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
            t1 = self.t1(m, u2, x)
            v1 = u1 * tf.exp(s1) + t1

            # Pre-compute network outputs for v2
            s2 = self.s2(m, v1, x)
            # Clamp s2 if specified
            if self.alpha is not None:
                s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
            t2 = self.t2(m, v1, x)
            v2 = u2 * tf.exp(s2) + t2
            v = tf.concat((v1, v2), axis=-1)

            if log_det_J:
                # log|J| = log(prod(diag(J))) -> according to inv architecture
                return v, tf.reduce_sum(s1, axis=-1) + tf.reduce_sum(s2, axis=-1)
            return v

        # --- Inverse pass --- #
        else:

            v1, v2 = tf.split(params, [self.n_out1, self.n_out2], axis=-1)

            # Pre-Compute s2
            s2 = self.s2(m, v1, x)
            # Clamp s2 if specified
            if self.alpha is not None:
                s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
            u2 = (v2 - self.t2(m, v1, x)) * tf.exp(-s2)

            # Pre-Compute s1
            s1 = self.s1(m, u2, x)
            # Clamp s1 if specified
            if self.alpha is not None:
                s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
            u1 = (v1 - self.t1(m, u2, x)) * tf.exp(-s1)
            u = tf.concat((u1, u2), axis=-1)

            if self.permutation is not None:
                u = self.permutation(u, inverse=True)
            return u


class InvariantBayesFlow(tf.keras.Model):
    """Implements a chain of conditional invertible blocks for Bayesian parameter inference."""

    def __init__(self, meta):
        """
        Creates a chain of cINN blocks and chains operations.
        ----------

        Arguments:
        meta        : list -- a list of dictionary, where each dictionary holds parameter - value pairs for a single
                                  keras.Dense layer
        """

        super(InvariantBayesFlow, self).__init__()

        self.cINNs = [ConditionalCouplingLayer(meta) for _ in range(meta['n_coupling_layers'])]
        self.z_dim = meta['n_params']
        self.n_models = meta['n_models']

    def call(self, m, params, x, inverse=False):
        """
        Performs one pass through an invertible chain (either inverse or forward).
        ----------

        Arguments:
        m         : tf.Tensor of shape (batch_size, n_models) -- the one-hot-encoded model indices
        params    : tf.Tensor of shape (batch_size, inp_dim) -- the parameters theta ~ p(theta|x) of interest
        x         : tf.Tensor of shape (batch_size, summary_dim) -- the conditional data x
        inverse   : bool -- flag indicating whether to tun the chain forward or backwards
        ----------

        Returns:
        (z, log_det_J)  :  (tf.Tensor of shape (batch_size, inp_dim), tf.Tensor of shape (batch_size, )) --
                           the transformed input, if inverse = False, and the corresponding Jacobian of the transformation
                            if inverse = False
        x               :  tf.Tensor of shape (batch_size, inp_dim) -- the transformed out, if inverse = True
        """

        if inverse:
            return self.inverse(m, params, x)
        else:
            return self.forward(m, params, x)

    def forward(self, m, params, x):
        """Performs a forward pass though the chain."""

        z = params
        log_det_Js = []
        for cINN in self.cINNs:
            z, log_det_J = cINN(m, z, x)
            log_det_Js.append(log_det_J)
        # Sum Jacobian determinants for all blocks to obtain total Jacobian.
        log_det_J = tf.add_n(log_det_Js)
        return z, log_det_J

    def inverse(self, m, z, x):
        """Performs a reverse pass through the chain."""

        params = z
        for cINN in reversed(self.cINNs):
            params = cINN(m, params, x, inverse=True)
        return params

    def sample(self, x, m, n_samples, to_numpy=True):
        """
        Samples from the inverse model given a single instance x.
        ----------

        Arguments:
        x         : tf.Tensor of shape (n_obs, x_dim) -- the conditioning data of interest
        m         : int - the integer model index
        n_samples : int -- number of samples to obtain from the approximate posterior
        to_numpy  : bool -- flag indicating whether to return the samples as a np.array or a tf.Tensor
        ----------

        Returns:
        theta_samples : 3D tf.Tensor or np.array of shape (n_samples, n_batch, theta_dim)
        """

        # Represent model index
        m_oh = tf.stack([tf.keras.utils.to_categorical(m, self.n_models)] * n_samples, axis=0)

        # Sample in parallel
        z_normal_samples = tf.random.normal(shape=(n_samples, self.z_dim), dtype=tf.float32)
        theta_samples = self.inverse(m_oh, z_normal_samples, tf.stack([x] * n_samples, axis=0))

        if to_numpy:
            return theta_samples.numpy()
        return theta_samples