import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2


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
        meta  : list -- a list of dictionaries, wherein each dictionary holds parameter - value pairs for a single
                       tf.keras.Dense layer.
        n_out : int  -- number of outputs of the coupling net
        """

        super(CouplingNet, self).__init__()

        self.dense = tf.keras.Sequential(
            # Hidden layer structure
            [tf.keras.layers.Dense(units,
                                   activation=meta['activation'],
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))
             for units in meta['n_units']] +
            # Output layer
            [tf.keras.layers.Dense(n_out,
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))]
        )

    def call(self, theta, x):
        """
        Concatenates x and y and performs a forward pass through the coupling net.
        Arguments:
        theta : tf.Tensor of shape (batch_size, inp_dim)     -- the parameters x ~ p(x|y) of interest
        x     : tf.Tensor of shape (batch_size, summary_dim) -- the summarized conditional data of interest y = sum(y)
        """

        inp = tf.concat((theta, x), axis=-1)
        out = self.dense(inp)
        return out


class ConditionalInvertibleBlock(tf.keras.Model):
    """Implements a conditional version of the INN block."""

    def __init__(self, meta, theta_dim, alpha=1.9, permute=False):
        """
        Creates a conditional invertible block.
        ----------

        Arguments:
        meta      : list -- a list of dictionaries, wherein each dictionary holds parameter - value pairs for a single
                       tf.keras.Dense layer. All coupling nets are assumed to be equal.
        theta_dim : int  -- the number of outputs of the invertible block (eq. the dimensionality of the latent space)
        """

        super(ConditionalInvertibleBlock, self).__init__()
        self.alpha = alpha
        self.n_out1 = theta_dim // 2
        self.n_out2 = theta_dim // 2 if theta_dim % 2 == 0 else theta_dim // 2 + 1
        if permute:
            self.permutation = Permutation(theta_dim)
        else:
            self.permutation = None
        self.s1 = CouplingNet(meta, self.n_out1)
        self.t1 = CouplingNet(meta, self.n_out1)
        self.s2 = CouplingNet(meta, self.n_out2)
        self.t2 = CouplingNet(meta, self.n_out2)

    def call(self, theta, x, inverse=False, log_det_J=True):
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
                theta = self.permutation(theta)

            u1, u2 = tf.split(theta, [self.n_out1, self.n_out2], axis=-1)

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

            v1, v2 = tf.split(theta, [self.n_out1, self.n_out2], axis=-1)

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


class BayesFlow(tf.keras.Model):
    """Implements a chain of conditional invertible blocks for Bayesian parameter inference."""

    def __init__(self, meta, n_blocks, theta_dim, alpha=1.9, summary_net=None, permute=False):
        """
        Creates a chain of cINN blocks and chains operations.
        ----------

        Arguments:
        meta        : list -- a list of dictionary, where each dictionary holds parameter - value pairs for a single
                                  keras.Dense layer
        n_blocks    : int  -- the number of invertible blocks
        theta_dim   : int  -- the dimensionality of the parameter space to be learned
        summary_net : tf.keras.Model or None -- an optinal summary network for learning the sumstats of x
        permute     : bool -- whether to permute the inputs to the cINN
        """

        super(BayesFlow, self).__init__()

        self.cINNs = [ConditionalInvertibleBlock(meta, theta_dim, alpha=alpha, permute=permute) for _ in range(n_blocks)]
        self.summary_net = summary_net
        self.theta_dim = theta_dim

    def call(self, theta, x, inverse=False):
        """
        Performs one pass through an invertible chain (either inverse or forward).
        ----------

        Arguments:
        theta     : tf.Tensor of shape (batch_size, inp_dim) -- the parameters theta ~ p(theta|x) of interest
        x         : tf.Tensor of shape (batch_size, summary_dim) -- the conditional data x
        inverse   : bool -- flag indicating whether to tun the chain forward or backwards
        ----------

        Returns:
        (z, log_det_J)  :  (tf.Tensor of shape (batch_size, inp_dim), tf.Tensor of shape (batch_size, )) --
                           the transformed input, if inverse = False, and the corresponding Jacobian of the transformation
                            if inverse = False
        x               :  tf.Tensor of shape (batch_size, inp_dim) -- the transformed out, if inverse = True
        """

        if self.summary_net is not None:
            x = self.summary_net(x)
        if inverse:
            return self.inverse(theta, x)
        else:
            return self.forward(theta, x)

    def forward(self, theta, x):
        """Performs a forward pass though the chain."""

        z = theta
        log_det_Js = []
        for cINN in self.cINNs:
            z, log_det_J = cINN(z, x)
            log_det_Js.append(log_det_J)
        # Sum Jacobian determinants for all blocks to obtain total Jacobian.
        log_det_J = tf.add_n(log_det_Js)
        return {'z': z, 'log_det_J': log_det_J}

    def inverse(self, z, x):
        """Performs a reverse pass through the chain."""

        theta = z
        for cINN in reversed(self.cINNs):
            theta = cINN(theta, x, inverse=True)
        return theta

    def sample(self, x, n_samples, to_numpy=False, training=False):
        """
        Samples from the inverse model given a single instance y or a batch of instances.
        ----------

        Arguments:
        x         : tf.Tensor of shape (batch_size, summary_dim) -- the conditioning data of interest
        n_samples : int -- number of samples to obtain from the approximate posterior
        to_numpy  : bool -- flag indicating whether to return the samples as a np.array or a tf.Tensor
        training  : bool -- flag used to indicate that samples are drawn are training time (BatchNorm)
        ----------

        Returns:
        theta_samples : 3D tf.Tensor or np.array of shape (n_samples, n_batch, theta_dim)
        """

        # Summarize obs data if summary net available
        if self.summary_net is not None:
            x = self.summary_net(x, training=training)

        # In case x is a single instance
        if int(x.shape[0]) == 1:
            z_normal_samples = tf.random_normal(shape=(n_samples, self.theta_dim), dtype=tf.float32)
            theta_samples = self.inverse(z_normal_samples, tf.tile(x, [n_samples, 1]))
        # In case of a batch input, send a 3D tensor through the invertible chain and use tensordot
        # Warning: This tensor could get pretty big if sampling a lot of values for a lot of batch instances!
        else:
            z_normal_samples = tf.random_normal(shape=(n_samples, int(x.shape[0]), self.theta_dim), dtype=tf.float32)
            theta_samples = self.inverse(z_normal_samples, tf.stack([x] * n_samples))

        if to_numpy:
            return theta_samples.numpy()
        return theta_samples


class InvariantModule(tf.keras.Model):
    """Implements an invariant nn module as proposed by Bloem-Reddy and Teh (2019)."""

    def __init__(self, meta):
        """
        Creates an invariant function with mean pooling.
        ----------

        Arguments:
        meta : dict -- a dictionary with hyperparameter name - values
        """

        super(InvariantModule, self).__init__()


        self.module = tf.keras.Sequential([
            tf.keras.layers.Dense(**meta['dense_inv_args'])
            for _ in range(meta['n_dense_inv'])
        ])

        self.weights_layer = tf.keras.Sequential([
                tf.keras.layers.Dense(**meta['dense_inv_args'])
                for _ in range(meta['n_dense_inv'])
            ] + 
            [
                tf.keras.layers.Dense(meta['dense_inv_args']['units'])
            
            ])
            

        self.post_pooling_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(**meta['dense_inv_args'])
            for _ in range(meta['n_dense_inv'])
        ])

    def call(self, x):
        """
        Transofrms the input into an invariant representation.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n, m) - the input where n is the 'time' or 'samples' dimensions
            over which pooling is performed and m is the input dimensionality
        ----------

        Returns:
        out : tf.Tensor of shape (batch_size, h_dim) -- the pooled and invariant representation of the input
        """

        # Embed
        x_emb = self.module(x)

        # Compute weights
        w = tf.nn.softmax(self.weights_layer(x), axis=1)
        w_x = tf.reduce_sum(x_emb * w, axis=1)
    
        # Increase representational power
        out = self.post_pooling_dense(w_x)
        return out


class EquivariantModule(tf.keras.Model):
    """Implements an equivariant nn module as proposed by Bloem-Reddy and Teh (2019)."""

    def __init__(self, meta):
        """
        Creates an equivariant neural network consisting of a FC network with
        equal number of hidden units in each layer and an invariant module
        with the same FC structure.
        ----------

        Arguments:
        meta : dict -- a dictionary with hyperparameter name - values
        """

        super(EquivariantModule, self).__init__()

        self.module = tf.keras.Sequential([
            tf.keras.layers.Dense(**meta['dense_equiv_args'])
            for _ in range(meta['n_dense_equiv'])
        ])

        self.invariant_module = InvariantModule(meta)

    def call(self, x):
        """
        Transofrms the input into an equivariant representation.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n, m) - the input where n is the 'time' or 'samples' dimensions
            over which pooling is performed and m is the input dimensionality
        ----------

        Returns:
        out : tf.Tensor of shape (batch_size, h_dim) -- the pooled and invariant representation of the input
        """

        x_inv = self.invariant_module(x)
        x_inv = tf.stack([x_inv] * int(x.shape[1]), axis=1) # Repeat x_inv n times
        x = tf.concat((x_inv, x), axis=-1)
        out = self.module(x)
        return out


class InvariantNetwork(tf.keras.Model):
    """
    Implements a network which parameterizes a
    permutationally invariant function according to Bloem-Reddy and Teh (2019).
    """

    def __init__(self, meta):
        """
        Creates a permutationally invariant network
        consisting of two equivariant modules and one invariant module.
        ----------

        Arguments:
        meta : dict -- hyperparameter settings for the equivariant and invariant modules
        """

        super(InvariantNetwork, self).__init__()

        self.equiv = tf.keras.Sequential([
            EquivariantModule(meta)
            for _ in range(meta['n_equiv'])
        ])
        self.inv = InvariantModule(meta)


    def call(self, x, **kwargs):
        """
        Transofrms the input into a permutationally invariant
        representation by first passing it through multiple equivariant
        modules in order to increase representational power.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n, m) - the input where n is the 'time' or
        'samples' dimensions over which pooling is performed and m is the input dimensionality
        ----------

        Returns:
        out : tf.Tensor of shape (batch_size, h_dim) -- the pooled and invariant representation of the input
        """

        x = self.equiv(x)
        out = self.inv(x)
        return out


class SequenceNetwork(tf.keras.Model):
    """
    Implements a network capable of dealing with sequence data.
    """

    def __init__(self, meta):
        """
        Creates a permutationally invariant network
        consisting of two equivariant modules and one invariant module.
        ----------

        Arguments:
        meta : dict -- hyperparameter settings for the equivariant and invariant modules
        """

        super(SequenceNetwork, self).__init__()
        self.lstm = tf.keras.layers.CuDNNLSTM(meta['lstm_units'], kernel_initializer='glorot_normal')
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, kernel_size=5, strides=1, activation='elu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, activation='elu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, activation='elu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Conv1D(128, kernel_size=2, strides=1, activation='elu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Conv1D(128, kernel_size=2, strides=1, activation='elu', kernel_initializer='glorot_normal'),
            tf.keras.layers.GlobalAveragePooling1D()
        ])


    def call(self, x, **kwargs):
        """
        Transofrms a sequence input into a fixed-size vector-representation.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n, m) - the input where n is the 'time' or
        'samples' dimensions over which pooling is performed and m is the input dimensionality
        ----------

        Returns:
        out : tf.Tensor of shape (batch_size, h_dim) -- the fixed size representation of th einput
        """


        out_lstm = self.lstm(x)
        out_conv = self.conv(x)
        out = tf.concat([out_lstm, out_conv], axis=-1)
        return out


class DeepEvidentialModel(tf.keras.Model):

    def __init__(self, meta, inv_xdim=False):
        super(DeepEvidentialModel, self).__init__()

        # A network to learn summary
        if meta['net_type'] == 'invariant':
            self.net = InvariantNetwork(meta)
        elif meta['net_type']  == 'sequence':
            self.net = SequenceNetwork(meta)
        else:
            raise NotImplementedError('net_type should be either of type "invariant" or "sequence"')

        # A network to increase representation power (post-pooling)
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(**meta['dense_post_args'])
            for _ in range(meta['n_dense_post'])
        ])

        # The layer to output model evidences
        self.evidence_layer = tf.keras.layers.Dense(meta['n_models'], activation='relu')
        self.M = meta['n_models']
        self.inv_xdim = inv_xdim

    def call(self, x):
        """
        Computes evidences for model selection given a batch of data.
        ----------

        Arguments:
        x            : tf.Tensor of shape (batch_size, n, m) -- the input where n is the 'time' or 'samples' dimensions
                        over which pooling is performed and m is the input dimensionality
        ----------

        Returns:
        alpha      : tf.Tensor of shape (batch_size, n_models) -- the model evidences
        alpha0     : tf.Tensor of shape (batch_size, 1) -- the Dirichlet strength
        m_probs    : tf.Tensor of shape (batch_size, n_models) -- the model posterior probabilities
        """


        # Compute evidence
        alpha = self.evidence(x)

        # Compute Dirichlet strength (alpha0) and mean (m_probs)
        alpha0 = tf.reduce_sum(alpha, axis=1, keepdims=True)
        m_probs = alpha / alpha0
        return {'alpha': alpha, 'alpha0': alpha0 ,'m_probs': m_probs}

    def compute_summary(self, x):
        """Returns the final representation before the evidence layer."""

        # Compute summary representation
        if self.inv_xdim:
            x = tf.split(x, int(x.shape[2]), axis=-1)
            x = tf.concat([self.net(x_dim) for x_dim in x], axis=-1)
        else:
            x = self.net(x)
        # Combine summary
        x = self.dense(x)
        return x

    def predict(self, x, to_numpy=True):
        """Returns the mean, variance and uncertainty of the Dirichlet distro."""

        alpha = self.evidence(x)
        alpha0 = tf.reduce_sum(alpha, axis=1, keepdims=True)
        mean = alpha / alpha0
        var = alpha * (alpha0 - alpha) / (alpha0 * alpha0 * (alpha0 + 1))
        uncertainty = self.M / alpha0

        if to_numpy:
            mean = mean.numpy()
            var = var.numpy()
            uncertainty = uncertainty.numpy()

        return {'m_probs': mean, 'm_var': var, 'uncertainty': uncertainty}

    def evidence(self, x):
        """Computes the evidence vector (alpha) of the Dirichlet distro."""

        # Summarize into fixed size
        x = self.compute_summary(x)

        # Compute eviddences
        evidence = self.evidence_layer(x)
        alpha = evidence + 1
        return alpha

    def sample(self, x, n_samples=5000, to_numpy=True):
        """
        Samples posterior model probabilities from the second-order Dirichlet distro.
        ----------

        Arguments:
        x         : tf.Tensor of shape (batch_size, n_points, n_features) -- the observed data
        n_samples : int -- number of samples to obtain from the approximate posterior (default 5000)
        to_numpy  : bool -- flag indicating whether to return the samples as a np.array or a tf.Tensor
        ----------

        Returns:
        pm_samples : tf.Tensor or np.array of shape (n_samples, n_batch, n_models) -- the posterior samples from the Dirichlet distro
        """

        # Compute evidential values
        ev = self(x)
        alpha = ev['alpha'].numpy()
        N = alpha.shape[0]

        # Sample for each dataset
        pm_samples = np.stack([np.random.dirichlet(alpha[n, :], size=n_samples) for n in range(N)], axis=1)

        if not to_numpy:
             pm_samples = tf.convert_to_tensor(pm_samples, dtype=tf.float32)
        return pm_samples


class VAE(tf.keras.Model):

    def __init__(self, meta):
        super(VAE, self).__init__()

        # Dimensions of latent space and number of models
        self.z_dim = meta['z_dim']
        self.M = meta['n_models']

        # Summary network
        if meta['summary_type'] == 'invariant':
            self.summary_net = InvariantNetwork(meta['summary_meta'])
        elif meta['summary_type']  == 'sequence':
            self.summary_net = SequenceNetwork(meta['summary_meta'])
        elif meta['summary_type'] is None:
            self.summary_net = None
        else:
            raise NotImplementedError('net_type should be either of type "invariant" or "sequence"')

        # Encoder network
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(**meta['encoder_dense_args'])
            for _ in range(meta['n_dense_encoder'])
        ])

        # Encoder output to z
        self.z_mapper = tf.keras.layers.Dense(meta['z_dim'] * 2)
        self.logits_layer = tf.keras.layers.Dense(meta['n_models'])

    def call(self, x, return_prob=False):
        """
        Computes a summary of h(y), concatenates x and h(y) and passes them through encoder and decoder.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n_obs, inp_dim)  -- the simulated batch of data
        m : tf.Tensor of shape (batch_size, num_models)      -- the one-hot encoded model indices
        return_probs : bool -- a flag ondicating whether to return logits or probabilities (softmax)

        ----------

        Output:
        z_mean   : tf.Tensor of shape (batch_size, z_dim) -- the means of the latent Gaussian distribution
        z_logvar : tf.Tensor of shape (batch_size, z_dim) -- the logvars of the latent Gaussian distribution
        """

        # Summarize (get fixed-size vector)
        if self.summary_net is not None:
            x = self.summary_net(x)

        # Encode into
        x = self.encoder(x)
        x = self.z_mapper(x)
        z_mean, z_logvar = tf.split(x, 2, axis=-1)

        # Sample
        eps = tf.random_normal(shape=z_mean.shape)
        z = z_mean + eps * tf.exp(z_logvar * 0.5)

        # Probabilistic classification
        m_logits = self.logits_layer(z)
        m_probs = tf.nn.softmax(m_logits, axis=1)
        return {'z_mean': z_mean,
                'z_logvar': z_logvar,
                'z_samples': z,
                'm_logits': m_logits,
                'm_probs': m_probs}

    def predict(self, x, to_numpy=True):
        """
        Returns approximate model posterior probabilities given a tensor dataset.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n_obs, inp_dim)  -- the simulated batch of data
        m : tf.Tensor of shape (batch_size, num_models)      -- the one-hot encoded model indices
        return_probs : bool -- a flag ondicating whether to return logits or probabilities (softmax)

        ----------

        Output:
        z_mean   : tf.Tensor of shape (batch_size, z_dim) -- the means of the latent Gaussian distribution
        """

        # Summarize (get fixed-size vector)
        if self.summary_net is not None:
            x = self.summary_net(x)

        # Get z
        x = self.encoder(x)
        x = self.z_mapper(x)
        z_mean, _ = tf.split(x, 2, axis=-1)

        # Decode mean of latent distribution
        m_logits = self.logits_layer(z_mean)
        m_probs = tf.nn.softmax(m_logits, axis=1)

        if to_numpy:
            m_probs = m_probs.numpy()
            m_logits = m_logits.numpy()

        return {'m_probs': m_probs, 'm_logits': m_logits}

    def sample(self, x, n_samples, to_numpy=True):
        """
        Samples from the decoder given a single instance y or a batch of instances.
        ----------

        Arguments:
        x         : tf.Tensor of shape (batch_size, n_points) -- the conditional data of interest
        n_samples : int -- number of samples to obtain from the approximate model posterior
        to_numpy  : bool -- flag indicating whether to return the samples as a np.array or a tf.Tensor
        ----------

        Returns:
        m_samples : 3D tf.Tensor or np.array of shape (n_samples, n_batch, n_models)
        """

        # Summarize (get fixed-size vector)
        if self.summary_net is not None:
            x = self.summary_net(x)
        x = self.encoder(x)

        # Get z
        x = self.z_mapper(x)
        z_mean, z_logvar = tf.split(x, 2, axis=-1)

        # Sample
        eps = tf.random_normal(shape=(n_samples, z_mean.shape[0], z_mean.shape[1]))
        z = z_mean + eps * tf.exp(z_logvar * 0.5)
        z = tf.transpose(z, [1, 0, 2])

        m_samples = tf.nn.softmax(self.logits_layer(z), axis=-1)
        m_samples = tf.transpose(m_samples, [1, 0, 2])
        if to_numpy:
            return m_samples.numpy()
        return m_samples


class MCDropOutModel(tf.keras.Model):
    """
    Implements a heteroscedastic classification model according to Kendal and Gal (2017).
    """
    def __init__(self, meta):
        super(MCDropOutModel, self).__init__()

        # Number of models and number of dropout samples
        self.M = meta['n_models']

        # Summary network
        if meta['summary_type'] == 'invariant':
            self.summary_net = InvariantNetwork(meta['summary_meta'])
        elif meta['summary_type']  == 'sequence':
            self.summary_net = SequenceNetwork(meta['summary_meta'])
        elif meta['summary_type'] is None:
            self.summary_net = None
        else:
            raise NotImplementedError('net_type should be either of type "invariant" or "sequence"')

        # A network to increase representation power (post-pooling)
        dense_layers = []
        for _ in range(meta['n_dense_post']):
            dense_layers.append(tf.keras.layers.Dense(**meta['dense_post_args']))
            dense_layers.append(tf.keras.layers.Dropout(meta['dropout_rate']))
        self.dense_net = tf.keras.Sequential(dense_layers)

        # Logits layers, i.e., fully connected with linear activation
        self.logits_layer = tf.keras.layers.Dense(meta['n_models'])

    def call(self, x):
        """
        Computes a summary h(x) and passes it through a feed-forward network.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n_obs, inp_dim)  -- the simulated batch of data

        ----------

        Output:
        m_hat tf.Tensor of shape (batch_size, dropout_samples, n_models) -- the MC logits samples
        """

        # Compute summary, if summary net has been given
        if self.summary_net is not None:
            x = self.summary_net(x)

        # Obtain logits
        x_l = self.dense_net(x, training=True)
        logits = self.logits_layer(x_l)
        m_probs = tf.nn.softmax(logits, axis=-1)
        return {'m_logits': logits, 'm_probs': m_probs}


    def predict(self, x, n_samples=1000, to_numpy=True):
        """
        Approximates model posterior probabilities given a tensor dataset.
        """

        # Compute summary, if summary net has been given
        if self.summary_net is not None:
            x = self.summary_net(x)

        # Create representation for parallelized dropout
        x = tf.stack([x] * n_samples, axis=1)

        # Obtain logits and probs
        x_l = self.dense_net(x, training=True)
        logits = self.logits_layer(x_l)
        m_samples = tf.nn.softmax(logits, axis=-1)
        m_probs = tf.reduce_mean(m_samples, axis=1)
        m_logits = tf.reduce_mean(logits, axis=1)

        if to_numpy:
            m_logits = m_logits.numpy()
            m_probs = m_probs.numpy()

        return {'m_logits': m_logits, 'm_probs': m_probs}

    def sample(self, x, n_samples, to_numpy=False):
        """
        Samples from the decoder given a single instance y or a batch of instances.
        ----------

        Arguments:
        x         : tf.Tensor of shape (batch_size, n_points) -- the conditional data of interest
        n_samples : int -- number of samples to obtain from the approximate model posterior
        to_numpy  : bool -- flag indicating whether to return the samples as a np.array or a tf.Tensor
        ----------

        Returns:
        m_samples : 3D tf.Tensor or np.array of shape (n_samples, n_batch, n_models)
        """

        # Compute summary, if summary net has been given
        if self.summary_net is not None:
            x = self.summary_net(x)

        # Create representation for parallelized dropout
        x = tf.stack([x] * n_samples, axis=1)

        # Obtain logits
        x_l = self.dense_net(x, training=True)
        logits = self.logits_layer(x_l)
        m_samples = tf.nn.softmax(logits, axis=-1)
        m_samples = tf.transpose(m_samples, [1, 0, 2])

        if to_numpy:
            return m_samples.numpy()
        return m_samples


class SoftmaxModel(tf.keras.Model):
    """
    Implements a heteroscedastic classification model according to Kendal and Gal (2017).
    """

    def __init__(self, meta):
        super(SoftmaxModel, self).__init__()

        # Number of models and number of dropout samples
        self.M = meta['n_models']

        # Summary network
        if meta['summary_type'] == 'invariant':
            self.summary_net = InvariantNetwork(meta['summary_meta'])
        elif meta['summary_type']  == 'sequence':
            self.summary_net = SequenceNetwork(meta['summary_meta'])
        elif meta['summary_type'] is None:
            self.summary_net = None
        else:
            raise NotImplementedError('net_type should be either of type "invariant" or "sequence"')

        # A network to increase representation power (post-pooling)
        dense_layers = []
        for _ in range(meta['n_dense_post']):
            dense_layers.append(tf.keras.layers.Dense(**meta['dense_post_args']))
        self.dense_net = tf.keras.Sequential(dense_layers)

        # Logits layers, i.e., fully connected with linear activation
        self.logits_layer = tf.keras.layers.Dense(meta['n_models'])

    def call(self, x):
        """
        Computes a summary h(x) and passes it through a feed-forward network.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n_obs, inp_dim)  -- the simulated batch of data

        ----------

        Output:
        m_hat tf.Tensor of shape (batch_size, dropout_samples, n_models) -- the MC logits samples
        """

        # Compute summary, if summary net has been given
        if self.summary_net is not None:
            x = self.summary_net(x)

        # Obtain logits
        x_l = self.dense_net(x, training=True)
        logits = self.logits_layer(x_l)
        m_probs = tf.nn.softmax(logits, axis=-1)
        return {'m_logits': logits, 'm_probs': m_probs}


    def predict(self, x, to_numpy=True):
        """
        Approximates model probabilities given a tensor dataset.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n_obs, inp_dim)  -- the simulated batch of data
        ----------
        """

        out = self(x)
        if to_numpy:
            out['m_logits'] = out['m_logits'].numpy()
            out['m_probs'] = out['m_probs'].numpy()
        return out

    def sample(self, x, n_samples=None, to_numpy=False):
        """
        Simply performs a forward pass through the network.
        ----------

        Arguments:
        x         : tf.Tensor of shape (batch_size, n_points) -- the conditional data of interest
        n_samples : int -- number of samples to obtain from the approximate model posterior (has no effect)
        to_numpy  : bool -- flag indicating whether to return the samples as a np.array or a tf.Tensor
        ----------

        Returns:
        m_samples : 3D tf.Tensor or np.array of shape (n_samples, n_batch, n_models)
        """

        # Compute summary, if summary net has been given
        m_samples = tf.expand_dims(self(x)['m_probs'], axis=0)

        if to_numpy:
            return m_samples.numpy()
        return m_samples
