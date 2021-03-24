import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



class RegressionNetwork(tf.keras.Model):
    """
    Implements a simple regression network with keras.
    """
    
    def __init__(self, meta, summary_net=None):
        super(RegressionNetwork, self).__init__()
        self.summary_net = summary_net
        self.net = Sequential(
            [Dense(u, activation=meta['activation']) for u in meta['units']] +
            [Dense(meta['n_params'])]
        )
        
    def call(self, x):
        """
        Performs the forward pass of the model.
        
        Args:
        x - tf.Tensor of shape (batch_size, sum_stats)
        
        Returns:
        out - tf.Tensor of shape (batch_size, predicted_params)
        """
        
        if self.summary_net is not None:
            x = self.summary_net(x)
        out = self.net(x)
        return out
    
    
class HeteroscedasticRegressionNetwork(tf.keras.Model):
    """
    Implements a simple regression network with keras.
    """
    
    def __init__(self, meta, summary_net=None):
        super(HeteroscedasticRegressionNetwork, self).__init__()
        self.summary_net = summary_net
        self.net = Sequential(
            [Dense(u, activation=meta['activation']) for u in meta['units']] +
            [Dense(meta['n_params'] * 2)]
        )
        
    def call(self, x):
        """
        Performs the forward pass of the model.
        
        Args:
        x - tf.Tensor of shape (batch_size, sum_stats)
        
        Returns:
        pred_means - tf.Tensor of shape (batch_size, n_params) : predicted posterior means
        pred_vars  - tf.Tensor of shape (batch_size, n_params) : predicted posterior variances
        """
        
        if self.summary_net is not None:
            x = self.summary_net(x)
        out = self.net(x)

        pred_means, pred_vars = tf.split(out, 2, axis=-1)
        pred_vars = tf.nn.softplus(pred_vars)
        return pred_means, pred_vars
    
    
class InvariantModule(tf.keras.Model):
    
    def __init__(self, meta):
        super(InvariantModule, self).__init__()
        
        self.s1 = Sequential([Dense(**meta['dense_s1_args']) for _ in range(meta['n_dense_s1'])])
        self.s2 = Sequential([Dense(**meta['dense_s2_args']) for _ in range(meta['n_dense_s2'])])
                    
    def call(self, x):
        """
        Performs the forward pass of a learnable invariant transform.
        
        Args:
        x - tf.Tensor of shape (batch_size, N, x_dim)
        
        Returns:
        out - tf.Tensor of shape (batch_size, out_dim)
        """
        
        x_reduced = tf.reduce_mean(self.s1(x), axis=1)
        out = self.s2(x_reduced)
        return out
    
    
class EquivariantModule(tf.keras.Model):
    
    def __init__(self, meta):
        super(EquivariantModule, self).__init__()
        
        self.invariant_module = InvariantModule(meta)
        self.s3 = Sequential([Dense(**meta['dense_s3_args']) for _ in range(meta['n_dense_s3'])])
                    
    def call(self, x):
        """
        Performs the forward pass of a learnable equivariant transform.
        
        Args:
        x - tf.Tensor of shape (batch_size, N, x_dim)
        
        Returns:
        out - tf.Tensor of shape (batch_size, N, equiv_dim)
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
    def __init__(self, meta):
        super(InvariantNetwork, self).__init__()
        
        self.equiv_seq = Sequential([EquivariantModule(meta) for _ in range(meta['n_equiv'])])
        self.inv = InvariantModule(meta)
    
    def call(self, x):
        """
        Performs the forward pass of a learnable deep invariant transformation
        consisting of a sequence of equivariant transforms followed by an invariant transform.
        
        Args:
        x - tf.Tensor of shape (batch_size, n_obs, data_dim)
        
        Returns:
        out - tf.Tensor of shape (batch_size, out_dim + 1)
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
    """Implements a permutation layer to permute the input dimensions of the cINN block."""

    def __init__(self, input_dim):
        """
        Creates a permutation layer for a conditional invertible block.
        ----------

        Arguments:
        input_dim  : int -- the dimensionality of the input to the c inv block.
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
        """Permutes the bach of an input."""

        if not inverse:
            return tf.transpose(tf.gather(tf.transpose(x), self.permutation))
        return tf.transpose(tf.gather(tf.transpose(x), self.inv_permutation))


class CouplingNet(tf.keras.Model):
    """Implements a conditional version of a sequential network."""

    def __init__(self, meta, n_out):
        """
        Creates a conditional coupling net (FC neural network).
        ----------

        Arguments:
        meta  : dict -- a dictionary which holds arguments for a dense layer.
        n_out : int  -- number of outputs of the coupling net
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
        """
        Concatenates x and y and performs a forward pass through the coupling net.
        Arguments:
        params : tf.Tensor of shape (batch_size, n_params//2) -- the split parameters theta ~ p(theta) of interest
        x      : tf.Tensor of shape (batch_size, summary_dim) -- the summarized conditional data of interest x = sum(x)
        """

        inp = tf.concat((params, x), axis=-1)
        out = self.dense(inp)
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
            
        self.s1 = CouplingNet(meta['s_args'], self.n_out1)
        self.t1 = CouplingNet(meta['t_args'], self.n_out1)
        self.s2 = CouplingNet(meta['s_args'], self.n_out2)
        self.t2 = CouplingNet(meta['t_args'], self.n_out2)

    def call(self, params, x, inverse=False, log_det_J=True):
        """
        Implements both directions of a conditional invertible block.
        ----------

        Arguments:
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

    def __init__(self, meta):
        """
        Creates a chain of cINN blocks and chains operations with an optional summary network.
         TODO: - Allow for generic base distributions
        ----------

        Arguments:
        meta        : list -- a list of dictionary, where each dictionary holds parameter - value pairs for a single
                                  keras.Dense layer
        summary_net : tf.keras.Model or None -- an optinal summary network for learning the sumstats of x
        """
        super(InvertibleNetwork, self).__init__()

        self.cINNs = [ConditionalCouplingLayer(meta) for _ in range(meta['n_coupling_layers'])]
        self.z_dim = meta['n_params']

    def call(self, params, x, inverse=False):
        """
        Performs one pass through an invertible chain (either inverse or forward).
        ----------

        Arguments:
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
        ----------

        Arguments:
        x         : tf.Tensor of shape (n_datasets, summary_dim) -- the conditioning data set(s) of interest
        n_samples : int -- number of samples to obtain from the approximate posterior
        to_numpy  : bool -- flag indicating whether to return the samples as a np.array or a tf.Tensor
        ----------

        Returns:
        theta_samples : 3D tf.Tensor or np.array of shape (n_samples, n_datasets, n_params)
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

    def __init__(self, meta, summary_net=None):
        """
        Creates an evidential network and couples it with an optional summary network.
        ----------

        Arguments:
        meta        : list -- a list of dictionary, where each dictionary holds parameter - value pairs for a single
                                  keras.Dense layer
        summary_net : tf.keras.Model or None -- an optinal summary network for learning the sumstats of x
        """
        super(EvidentialNetwork, self).__init__()

        self.summary_net = summary_net

        # A network to increase representation power (post-pooling)
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(**meta['dense_args'])
            for _ in range(meta['n_dense'])
        ])

        # The layer to output model evidences
        self.evidence_layer = tf.keras.layers.Dense(meta['n_models'], activation=meta['out_activation'])
        self.J = meta['n_models']

    def call(self, sim_data):
        """
        Computes evidences for model comparison given a batch of data.
        ----------

        Arguments:
        sim_data   : tf.Tensor of shape (batch_size, n_obs, data_dim) -- the input where n_obs is the 'time' or 'samples' dimensions
                        over which pooling is performed and data_dim is the intrinsic input dimensionality
        ----------

        Returns:
        alpha      : tf.Tensor of shape (batch_size, n_models) -- the model evidences
        """

        # Compute evidence
        return self.evidence(sim_data)

    def compute_summary(self, sim_data):
        """
        Returns the final representation before the evidence layer.
        """

        # Summarize obs data if summary net available
        if self.summary_net is not None:
            sim_data = self.summary_net(sim_data)
        return sim_data

    def predict(self, obs_data, to_numpy=True):
        """
        Returns the mean, variance and uncertainty implied by the estimated Dirichlet density.
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
        """
        Computes the evidence vector (alpha + 1) as derived from the estimated Dirichlet density.
        """

        # Summarize into fixed size, if specified
        x = self.compute_summary(x)

        # Pass through dense layer
        x = self.dense(x)

        # Compute eviddences
        evidence = self.evidence_layer(x)
        alpha = evidence + 1
        return alpha

    def sample(self, obs_data, n_samples, to_numpy=True):
        """
        Samples posterior model probabilities from the second-order Dirichlet distro.
        ----------

        Arguments:
        obs_data  : tf.Tensor of shape (n_datasets, n_obs, data_dim) -- the actually observed (or simulated) data
        n_samples : int -- number of samples to obtain from the approximate posterior (default 5000)
        to_numpy  : bool -- flag indicating whether to return the samples as a np.array or a tf.Tensor
        ----------

        Returns:
        pm_samples : tf.Tensor or np.array of shape (n_samples, n_batch, n_models) -- the posterior samples from the Dirichlet distro
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
