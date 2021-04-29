import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

from bayesflow import default_settings
from bayesflow.helpers import build_meta_dict


class RegressionNetwork(tf.keras.Model):
    """Implements a simple regression network with keras.
    """
    
    def __init__(self, meta, summary_net=None):
        super(RegressionNetwork, self).__init__()
        self.summary_net = summary_net
        self.net = Sequential(
            [Dense(u, activation=meta['activation']) for u in meta['units']] +
            [Dense(meta['n_params'])]
        )
        
    def call(self, x):
        """Performs the forward pass of the model.
        
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, sum_stats)
        
        Returns
        -------
        out : tf.Tensor
         Output of shape (batch_size, predicted_params)
        """
        
        if self.summary_net is not None:
            x = self.summary_net(x)
        out = self.net(x)
        return out
    
    
class HeteroscedasticRegressionNetwork(tf.keras.Model):
    """Implements a simple regression network with keras.
    """
    
    def __init__(self, meta, summary_net=None):
        super(HeteroscedasticRegressionNetwork, self).__init__()
        self.summary_net = summary_net
        self.net = Sequential(
            [Dense(u, activation=meta['activation']) for u in meta['units']] +
            [Dense(meta['n_params'] * 2)]
        )
        
    def call(self, x):
        """ Performs the forward pass of the model.
        
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, sum_stats)
        
        Returns
        -------
        pred_means : tf.Tensor
            Predicted posterior means, shape (batch_size, n_params)
        pred_vars  : tf.Tensor
            Predicted posterior variances, shape (batch_size, n_params)
        """
        
        if self.summary_net is not None:
            x = self.summary_net(x)
        out = self.net(x)

        pred_means, pred_vars = tf.split(out, 2, axis=-1)
        pred_vars = tf.nn.softplus(pred_vars)
        return pred_means, pred_vars
    
    
class InvariantModule(tf.keras.Model):
    """Implements an invariant module with keras."""
    
    def __init__(self, meta):
        super(InvariantModule, self).__init__()
        
        self.s1 = Sequential([Dense(**meta['dense_s1_args']) for _ in range(meta['n_dense_s1'])])
        self.s2 = Sequential([Dense(**meta['dense_s2_args']) for _ in range(meta['n_dense_s2'])])
                    
    def call(self, x):
        """Performs the forward pass of a learnable invariant transform.
        
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, N, x_dim)
        
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, out_dim)
        """
        
        x_reduced = tf.reduce_mean(self.s1(x), axis=1)
        out = self.s2(x_reduced)
        return out
    
    
class EquivariantModule(tf.keras.Model):
    """Implements an equivariant module with keras."""
    
    def __init__(self, meta):
        super(EquivariantModule, self).__init__()
        
        self.invariant_module = InvariantModule(meta)
        self.s3 = Sequential([Dense(**meta['dense_s3_args']) for _ in range(meta['n_dense_s3'])])
                    
    def call(self, x):
        """Performs the forward pass of a learnable equivariant transform.
        
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, N, x_dim)
        
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, N, equiv_dim)
        """
        
        # Store N
        N = int(x.shape[1])
        
        # Output dim is (batch_size, inv_dim) - > (batch_size, N, inv_dim)
        out_inv = self.invariant_module(x)
        out_inv_rep = tf.stack([out_inv] * N, axis=1)
        
        # Concatenate each x with the repeated invariant embedding
        out_c = tf.concat([x, out_inv_rep], axis=-1)
        
        # Pass through equivariant func
        out = self.s3(out_c)
        return out


class InvariantNetwork(tf.keras.Model):
    """Implements an invariant network with keras.
    """

    def __init__(self, meta={}):
        super(InvariantNetwork, self).__init__()

        meta = build_meta_dict(user_dict=meta,
                               default_setting=default_settings.DEFAULT_SETTING_INVARIANT_NET)
        
        self.equiv_seq = Sequential([EquivariantModule(meta) for _ in range(meta['n_equiv'])])
        self.inv = InvariantModule(meta)
    
    def call(self, x):
        """ Performs the forward pass of a learnable deep invariant transformation consisting of
        a sequence of equivariant transforms followed by an invariant transform.
        
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, n_obs, data_dim)
        
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, out_dim + 1)
        """
        
        # Extract n_obs and create sqrt(N) vector
        N = int(x.shape[1])
        N_rep = tf.math.sqrt(N * tf.ones((x.shape[0], 1)))

        # Pass through series of augmented equivariant transforms
        out_equiv = self.equiv_seq(x)

        # Pass through final invariant layer and concatenate with N_rep
        out_inv = self.inv(out_equiv)
        out = tf.concat((out_inv, N_rep), axis=-1)

        return out
    
    
class Permutation(tf.keras.Model):
    """Implements a permutation layer to permute the input dimensions of the cINN block.
    """

    def __init__(self, input_dim):
        """ Creates a permutation layer for a conditional invertible block.


        Arguments
        ---------
        input_dim  : int
            Ihe dimensionality of the input to the c inv block.
        """

        super(Permutation, self).__init__()

        permutation_vec = np.random.permutation(input_dim)
        inv_permutation_vec = np.argsort(permutation_vec)
        self.permutation = tf.Variable(initial_value=permutation_vec,
                                       trainable=False,
                                       dtype=tf.int32,
                                       name='permutation')
        self.inv_permutation = tf.Variable(initial_value=inv_permutation_vec,
                                           trainable=False,
                                           dtype=tf.int32,
                                           name='inv_permutation')

    def call(self, x, inverse=False):
        """ Permutes the batch of an input.

        Parameters
        ----------
        x: tf.Tensor
            Input to the layer.
        inverse: bool, default: False
            Controls if the current pass is forward (``inverse=False``) or inverse (``inverse=True``).

        Returns
        -------
        out: tf.Tensor
            Permuted input

        """

        if not inverse:
            return tf.transpose(tf.gather(tf.transpose(x), self.permutation))
        return tf.transpose(tf.gather(tf.transpose(x), self.inv_permutation))


class CouplingNet(tf.keras.Model):
    """Implements a conditional version of a sequential network."""

    def __init__(self, meta, n_out):
        """Creates a conditional coupling net (FC neural network).

        Parameters
        ----------
        meta  : dict
            A dictionary which holds arguments for a dense layer.
        n_out : int
            Number of outputs of the coupling net
        """

        super(CouplingNet, self).__init__()

        self.dense = Sequential(
            # Hidden layer structure
            [Dense(units, 
                   activation=meta['activation'], 
                   kernel_initializer=meta['initializer'])
             for units in meta['units']] +
            # Output layer
            [Dense(n_out, kernel_initializer=meta['initializer'])]
        )

    def call(self, params, x):
        """Concatenates x and y and performs a forward pass through the coupling net.

        Parameters
        ----------
        params : tf.Tensor
          The split parameters :math:`\\theta \sim p(\\theta)` of interest, shape (batch_size, n_params//2)
        x      : tf.Tensor
            the summarized conditional data of interest ``x = summary(x)``, shape (batch_size, summary_dim)
        """

        inp = tf.concat((params, x), axis=-1)
        out = self.dense(inp)
        return out


class ConditionalCouplingLayer(tf.keras.Model):
    """Implements a conditional version of the INN block."""

    def __init__(self, meta):
        """Creates a conditional invertible block.

        Parameters
        ----------
        meta      : list(dict)
            A list of dictionaries, wherein each dictionary holds parameter-value pairs for a single
            :class:`tf.keras.Dense` layer. All coupling nets are assumed to be equal.
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
            
        self.s1 = CouplingNet(meta['s_args'], self.n_out1)
        self.t1 = CouplingNet(meta['t_args'], self.n_out1)
        self.s2 = CouplingNet(meta['s_args'], self.n_out2)
        self.t2 = CouplingNet(meta['t_args'], self.n_out2)

    def call(self, params, x, inverse=False, log_det_J=True):
        """Implements both directions of a conditional invertible block.

        Parameters
        ----------
        params     : tf.Tensor
            the parameters theta ~ p(theta|y) of interest, shape (batch_size, theta_dim) --
        x         : tf.Tensor
            the summarized conditional data of interest x = summary(x), shape (batch_size, summary_dim)
        inverse   : bool, default: False
            Flag indicating whether to run the block forward or backwards
        log_det_J : bool, default: True
            Flag indicating whether to return the log determinant of the Jacobian matrix

        Returns
        -------
        (v, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            v shape: (batch_size, inp_dim), log_det_J shape: (batch_size, )

        u               :  tf.Tensor
            If inverse=True: The transformed out, shape (batch_size, inp_dim)

        Important
        ---------
        If ``inverse=False``, the return is ``(v, log_det_J)``.\n
        If ``inverse=True``, the return is ``u``.
        """

        # --- Forward pass --- #
        if not inverse:

            if self.permutation is not None:
                params = self.permutation(params)

            u1, u2 = tf.split(params, [self.n_out1, self.n_out2], axis=-1)

            # Pre-compute network outputs for v1
            s1 = self.s1(u2, x)
            # Clamp s1 if specified
            if self.alpha is not None:
                s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
            t1 = self.t1(u2, x)
            v1 = u1 * tf.exp(s1) + t1

            # Pre-compute network outputs for v2
            s2 = self.s2(v1, x)
            # Clamp s2 if specified
            if self.alpha is not None:
                s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
            t2 = self.t2(v1, x)
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
            s2 = self.s2(v1, x)
            # Clamp s2 if specified
            if self.alpha is not None:
                s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
            u2 = (v2 - self.t2(v1, x)) * tf.exp(-s2)

            # Pre-Compute s1
            s1 = self.s1(u2, x)
            # Clamp s1 if specified
            if self.alpha is not None:
                s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
            u1 = (v1 - self.t1(u2, x)) * tf.exp(-s1)
            u = tf.concat((u1, u2), axis=-1)

            if self.permutation is not None:
                u = self.permutation(u, inverse=True)
            return u


class InvertibleNetwork(tf.keras.Model):
    """Implements a chain of conditional invertible blocks for Bayesian parameter inference."""

    def __init__(self, meta={}):
        """ Creates a chain of cINN blocks and chains operations with an optional summary network.

        Parameters
        ----------
        meta : list(dict)
            A list of dictionaries, where each dictionary holds parameter-value pairs
            for a single :class:`keras.Dense` layer

        Notes
        -----
        TODO: Allow for generic base distributions
        """
        super(InvertibleNetwork, self).__init__()

        meta = build_meta_dict(user_dict=meta,
                               default_setting=default_settings.DEFAULT_SETTING_INVERTIBLE_NET)

        self.cINNs = [ConditionalCouplingLayer(meta) for _ in range(meta['n_coupling_layers'])]
        self.z_dim = meta['n_params']

    def call(self, params, x, inverse=False):
        """Performs one pass through an invertible chain (either inverse or forward).

        Parameters
        ----------
        params    : tf.Tensor
            The parameters theta ~ p(theta|x) of interest, shape (batch_size, inp_dim)
        x         : tf.Tensor
            The conditional data x, shape (batch_size, summary_dim)
        inverse   : bool, default: False
            Flag indicating whether to run the chain forward or backwards

        Returns
        -------
        (v, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            v shape: (batch_size, inp_dim), log_det_J shape: (batch_size, )

        u               :  tf.Tensor
            If inverse=True: The transformed out, shape (batch_size, inp_dim)

        Important
        ---------
        If ``inverse=False``, the return is ``(v, log_det_J)``.\n
        If ``inverse=True``, the return is ``u``.
        """
        
        if inverse:
            return self.inverse(params, x)
        else:
            return self.forward(params, x)

    def forward(self, params, x):
        """Performs a forward pass though the chain."""

        z = params
        log_det_Js = []
        for cINN in self.cINNs:
            z, log_det_J = cINN(z, x)
            log_det_Js.append(log_det_J)
        # Sum Jacobian determinants for all blocks to obtain total Jacobian.
        log_det_J = tf.add_n(log_det_Js)
        return z, log_det_J

    def inverse(self, z, x):
        """Performs a reverse pass through the chain."""

        params = z
        for cINN in reversed(self.cINNs):
            params = cINN(params, x, inverse=True)
        return params

    def sample(self, x, n_samples, to_numpy=True):
        """
        Samples from the inverse model given a single instance y or a batch of instances.

        Parameters
        ----------
        x         : tf.Tensor
            The conditioning data set(s) of interest, shape (n_datasets, summary_dim)
        n_samples : int
            Number of samples to obtain from the approximate posterior
        to_numpy  : bool, default: True
            Flag indicating whether to return the samples as a `np.array` or a `tf.Tensor`

        Returns
        -------
        theta_samples : tf.Tensor or np.array
            Parameter samples, shape (n_samples, n_datasets, n_params)
        """

        # In case x is a single instance
        if int(x.shape[0]) == 1:
            z_normal_samples = tf.random.normal(shape=(n_samples, self.z_dim), dtype=tf.float32)
            param_samples = self.inverse(z_normal_samples, tf.tile(x, [n_samples, 1]))
        # In case of a batch input, send a 3D tensor through the invertible chain and use tensordot
        # Warning: This tensor could get pretty big if sampling a lot of values for a lot of batch instances!
        else:
            z_normal_samples = tf.random.normal(shape=(n_samples, int(x.shape[0]), self.z_dim), dtype=tf.float32)
            param_samples = self.inverse(z_normal_samples, tf.stack([x] * n_samples))

        if to_numpy:
            return param_samples.numpy()
        return param_samples


class EvidentialNetwork(tf.keras.Model):

    def __init__(self, meta):
        """Creates an evidential network and couples it with an optional summary network.

        Parameters
        ----------
        meta        : list(dict)
            A list of dictionaries, where each dictionary holds parameter-value pairs
            for a single :class:`tf.keras.Dense` layer
        """

        super(EvidentialNetwork, self).__init__()

        # A network to increase representation power (post-pooling)
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(**meta['dense_args'])
            for _ in range(meta['n_dense'])
        ])

        # The layer to output model evidences
        self.evidence_layer = tf.keras.layers.Dense(meta['n_models'], activation=meta['out_activation'])
        self.J = meta['n_models']

    def call(self, sim_data):
        """Computes evidences for model comparison given a batch of data.

        Parameters
        ----------
        sim_data   : tf.Tensor
            The input where `n_obs` is the ``time`` or ``samples`` dimensions over which pooling is
            performed and ``data_dim`` is the intrinsic input dimensionality, shape (batch_size, n_obs, data_dim)

        Returns
        -------
        alpha      : tf.Tensor of shape (batch_size, n_models) -- the model evidences
        """

        # Compute and return evidence
        return self.evidence(sim_data)

    def predict(self, obs_data, to_numpy=True):
        """Returns the mean, variance and uncertainty implied by the estimated Dirichlet density.

        Parameters
        ----------
        obs_data: tf.Tensor
            Observed data
        to_numpy: bool, default: True
            Flag that controls whether the output is a np.array or tf.Tensor

        Returns
        -------
        out: dict
            Dictionary with keys {m_probs, m_var, uncertainty}
        """

        alpha = self.evidence(obs_data)
        alpha0 = tf.reduce_sum(alpha, axis=1, keepdims=True)
        mean = alpha / alpha0
        var = alpha * (alpha0 - alpha) / (alpha0 * alpha0 * (alpha0 + 1))
        uncertainty = self.J / alpha0

        if to_numpy:
            mean = mean.numpy()
            var = var.numpy()
            uncertainty = uncertainty.numpy()

        return {'m_probs': mean, 'm_var': var, 'uncertainty': uncertainty}

    def evidence(self, x):
        """Computes the evidence vector (alpha + 1) as derived from the estimated Dirichlet density.

        Parameters
        ----------
        x  : tf.Tensor
            The conditional data set(s), shape (n_datasets, summary_dim)
        """

        # Pass through dense layer
        x = self.dense(x)

        # Compute evidences
        evidence = self.evidence_layer(x)
        alpha = evidence + 1
        return alpha

    def sample(self, obs_data, n_samples, to_numpy=True):
        """Samples posterior model probabilities from the second-order Dirichlet distro.

        Parameters
        ----------
        obs_data  : tf.Tensor
            The summary of the observed (or simulated) data, shape (n_datasets, summary_dim)
        n_samples : int
            Number of samples to obtain from the approximate posterior
        to_numpy  : bool, default: True
            Flag indicating whether to return the samples as a np.array or a tf.Tensor

        Returns
        -------
        pm_samples : tf.Tensor or np.array
            The posterior samples from the Dirichlet distribution, shape (n_samples, n_batch, n_models)
        """

        # Compute evidential values
        alpha = self.evidence(obs_data)
        n_datasets = alpha.shape[0]

        # Sample for each dataset
        pm_samples = np.stack([np.random.dirichlet(alpha[n, :], size=n_samples) for n in range(n_datasets)], axis=1)

        # Convert to tensor, if specified
        if not to_numpy:
             pm_samples = tf.convert_to_tensor(pm_samples, dtype=tf.float32)
        return pm_samples


class SequenceNet(tf.keras.Model):

    def __init__(self):
        """Creates a custom summary network, a combination of 1D conv and LSTM.
        """
        super(SequenceNet, self).__init__()

        self.conv_part = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 3, 3, activation='elu'),
            tf.keras.layers.Conv1D(64, 3, 3, activation='elu'),
            tf.keras.layers.Conv1D(64, 3, 3, activation='elu'),
            tf.keras.layers.GlobalAveragePooling1D()
        ])

        self.lstm_part = Sequential(
            [LSTM(32, return_sequences=True),
             LSTM(64)
             ])

    def call(self, x):
        """Performs a forward pass."""

        conv_out = self.conv_part(x)
        lstm_out = self.lstm_part(x)
        out = tf.concat((conv_out, lstm_out), axis=-1)
        return out
