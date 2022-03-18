import numpy as np
from scipy.stats import multivariate_t

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

from bayesflow import default_settings
from bayesflow.helpers import build_meta_dict
from bayesflow.exceptions import ConfigurationError


class TailNetwork(tf.keras.Model):
    
    def __init__(self, meta):
        """ Creates a network which will adaptively learn the heavy-tailedness of the target distribution.

        Parameters
        ----------
        meta : list(dict)
            A list of dictionaries, where each dictionary holds parameter-value pairs
            for a single :class:`keras.Dense` layer

        """
        
        super(TailNetwork, self).__init__()
        
        self.net = Sequential(
            [Dense(units, 
                   activation=meta['activation'], 
                   kernel_initializer=meta['initializer'])
             for units in meta['units']] +
            [Dense(1, kernel_initializer=meta['initializer'], activation='relu')]
        )
        
    def call(self, condition):
        """Performs a forward pass through the tail net. Output is the learned 'degrees of freedom' parameter
        for the latent t-distribution.

        Parameters
        ----------
        condition   : tf.Tensor
            the conditioning vector of interest, for instance ``x = summary(x)``, shape (batch_size, summary_dim)
        """
        # Output is bounded between (1, inf)
        out = self.net(condition) + 1.0
        return out

    
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

    def call(self, target, inverse=False):
        """ Permutes the batch of an input.

        Parameters
        ----------
        target   : tf.Tensor
            The vector to be permuted.
        inverse  : bool, default: False
            Controls if the current pass is forward (``inverse=False``) or inverse (``inverse=True``).

        Returns
        -------
        out      : tf.Tensor
            Permuted input

        """

        if not inverse:
            return tf.transpose(tf.gather(tf.transpose(target), self.permutation))
        return tf.transpose(tf.gather(tf.transpose(target), self.inv_permutation))


class ActNorm(tf.keras.Model):
    """Implements an Activation Normalization (ActNorm) Layer."""

    def __init__ (self, meta):
        """ Creates an instance of an ActNorm Layer as proposed by [1].

        Activation Normalization is learned invertible normalization, using
        a Scale (s) and Bias (b) vector [1].
            y = s * x + b (forward)
            x = (y - b)/s (inverse)
        
        The scale and bias can be data dependent initalized, such that the
        output has a mean of zero and standard deviation of one [1,2]. 
        Alternatively, it is initialized with vectors of ones (scale) and 
        zeros (bias).

        [1] - Kingma, Diederik P., and Prafulla Dhariwal. 
              "Glow: Generative flow with invertible 1x1 convolutions." 
               arXiv preprint arXiv:1807.03039 (2018).

        [2] - Salimans, Tim, and Durk P. Kingma. 
              "Weight normalization: A simple reparameterization to accelerate 
               training of deep neural networks." 
              Advances in neural information processing systems 29 
              (2016): 901-909.

        Parameters
        ----------
        meta : dict
            Contains initialization settings for the act norm layer.
        """

        super(ActNorm, self).__init__()
        # Initialize scale and bias with zeros and ones if no batch for initalization was provided.
        if meta.get('act_norm_init') is None:
            self.scale = tf.Variable(tf.ones((meta['n_params'], )),
                                     trainable=True,
                                     name='act_norm_scale')

            self.bias  = tf.Variable(tf.zeros((meta['n_params'], )),
                                     trainable=True,
                                     name='act_norm_bias')
        else:
            self._initalize_parameters_data_dependent(meta['act_norm_init'])

    def _initalize_parameters_data_dependent(self, init_data):
        """ Performs a data dependent initalization of the scale and bias.
        
        Initalizes the scale and bias vector as proposed by [1], such that the 
        layer output has a mean of zero and a standard deviation of one.

        Parameters
        ----------
        init_data : tf.Tensor
            of shape (batch size, number of parameters) to initialize
            the scale bias parameter by computing the mean and standard
            deviation along the first dimension of the Tensor.
        
        Returns
        -------
        (scale, bias) : tuple(tf.Tensor, tf.Tensor)
            scale and bias vector of shape (1, n_params).
        
        [1] - Salimans, Tim, and Durk P. Kingma. 
              "Weight normalization: A simple reparameterization to accelerate 
               training of deep neural networks." 
              Advances in neural information processing systems 29 
              (2016): 901-909.
        """
        
        # 2D Tensor case, assume first batch dimension
        if len(init_data.shape) == 2:
            mean = tf.math.reduce_mean(init_data, axis=0) 
            std  = tf.math.reduce_std(init_data,  axis=0)
        # 3D Tensor case, assume first batch dimension, second number of observations dimension
        elif len(init_data.shape) == 3:
            mean = tf.math.reduce_mean(init_data, axis=(0, 1)) 
            std  = tf.math.reduce_std(init_data,  axis=(0, 1))
        # Raise other cases
        else:
            raise ConfigurationError("""Currently, ActNorm supports only 2D and 3D Tensors, 
                                     but act_norm_init contains data with shape.""".format(init_data.shape))

        scale = 1.0 / std
        bias  = (-1.0 * mean) / std
        
        self.scale = tf.Variable(scale, trainable=True, name='act_norm_scale')
        self.bias  = tf.Variable(bias, trainable=True, name='act_norm_bias')

    def call(self, target, inverse=False):
        """ Performs one pass through the actnorm layer (either inverse or forward).
        
        Parameters
        ----------
        target     : tf.Tensor
            the target variables of interest, i.e., parameters for posterior estimation
        inverse    : bool, default: False
            Flag indicating whether to run the block forward or backwards
        
        Returns
        -------
        (z, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            v shape: (batch_size, inp_dim), log_det_J shape: (,)

        target          :  tf.Tensor
            If inverse=True: The inversly transformed targets, shape == target.shape

        Important
        ---------
        If ``inverse=False``, the return is ``(z, log_det_J)``.\n
        If ``inverse=True``, the return is ``target``.
        """
        
        if not inverse:
            return self._forward(target)
        else:
            return self._inverse(target)

    def _forward(self, target):
        """Performs a forward pass through the ActNorm layer."""

        z = self.scale * target + self.bias
        ldj = tf.math.reduce_sum(tf.math.log(tf.math.abs(self.scale)))
        if len(target.shape) == 3:
            ldj = ldj * target.shape[1]
        return z, ldj     

    def _inverse(self, target):
        """Performs an inverse pass through the ActNorm layer."""

        return (target - self.bias) / self.scale


class CouplingNet(tf.keras.Model):
    """Implements a conditional version of a satndard fully connected (FC) network."""

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

    def call(self, target, condition):
        """Concatenates target and condition and performs a forward pass through the coupling net.

        Parameters
        ----------
        target      : tf.Tensor
          The split estimation quntities, for instance, parameters :math:`\\theta \sim p(\\theta)` of interest, shape (batch_size, ...)
        condition   : tf.Tensor
            the conditioning vector of interest, for instance ``x = summary(x)``, shape (batch_size, summary_dim)
        """

        # Handle 3D case for a set-flow
        if len(target.shape) == 3:
            # Extract information about N
            N = int(target.shape[1])
            condition = tf.stack([condition] * N, axis=1)
        inp = tf.concat((target, condition), axis=-1)
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

        # Coupling net initialization
        self.alpha = meta['alpha']
        theta_dim = meta['n_params']
        self.n_out1 = theta_dim // 2
        self.n_out2 = theta_dim // 2 if theta_dim % 2 == 0 else theta_dim // 2 + 1
            
        self.s1 = CouplingNet(meta['s_args'], self.n_out1)
        self.t1 = CouplingNet(meta['t_args'], self.n_out1)
        self.s2 = CouplingNet(meta['s_args'], self.n_out2)
        self.t2 = CouplingNet(meta['t_args'], self.n_out2)

        # Optional permutation
        if meta['use_permutation']:
            self.permutation = Permutation(theta_dim)
        else:
            self.permutation = None

        # Optional activation normalization
        if meta['use_act_norm']:
            self.act_norm = ActNorm(meta)
        else:
            self.act_norm = None

    def _forward(self, target, condition):
        """ Performs a forward pass through the coupling block. Used internally by the instance.

        Parameters
        ----------
        target     : tf.Tensor
            the estimation quantities of interest, for instance, parameter vector of shape (batch_size, theta_dim)
        condition  : tf.Tensor
            the conditioning vector of interest, for instance, x = summary(x), shape (batch_size, summary_dim)

        Returns
        -------
        (v, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            v shape: (batch_size, inp_dim), log_det_J shape: (batch_size, )
        """

        # Split parameter vector
        u1, u2 = tf.split(target, [self.n_out1, self.n_out2], axis=-1)

        # Pre-compute network outputs for v1
        s1 = self.s1(u2, condition)
        # Clamp s1 if specified
        if self.alpha is not None:
            s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
        t1 = self.t1(u2, condition)
        v1 = u1 * tf.exp(s1) + t1

        # Pre-compute network outputs for v2
        s2 = self.s2(v1, condition)
        # Clamp s2 if specified
        if self.alpha is not None:
            s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
        t2 = self.t2(v1, condition)
        v2 = u2 * tf.exp(s2) + t2
        v = tf.concat((v1, v2), axis=-1)

        # Compute ldj, # log|J| = log(prod(diag(J))) -> according to inv architecture
        log_det_J = tf.reduce_sum(s1, axis=-1) + tf.reduce_sum(s2, axis=-1)
        return v, log_det_J 

    def _inverse(self, z, condition):
        """ Performs an inverse pass through the coupling block. Used internally by the instance.

        Parameters
        ----------
        z         : tf.Tensor
            latent variables z ~ p(z), shape (batch_size, theta_dim)
        condition  : tf.Tensor
            the conditioning vector of interest, for instance, x = summary(x), shape (batch_size, summary_dim)

        Returns
        -------
        (v, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            v shape: (batch_size, inp_dim), log_det_J shape: (batch_size, )
        """

        v1, v2 = tf.split(z, [self.n_out1, self.n_out2], axis=-1)

        # Pre-Compute s2
        s2 = self.s2(v1, condition)
        # Clamp s2 if specified
        if self.alpha is not None:
            s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
        u2 = (v2 - self.t2(v1, condition)) * tf.exp(-s2)

        # Pre-Compute s1
        s1 = self.s1(u2, condition)
        # Clamp s1 if specified
        if self.alpha is not None:
            s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
        u1 = (v1 - self.t1(u2, condition)) * tf.exp(-s1)
        u = tf.concat((u1, u2), axis=-1)

        return u

    def call(self, target_or_z, condition, inverse=False):
        """Performs one pass through an invertible chain (either inverse or forward).
        
        Parameters
        ----------
        target_or_z      : tf.Tensor
            the estimation quantites of interest or latent representations z ~ p(z), shape (batch_size, ...)
        condition        : tf.Tensor
            the conditioning data of interest, for instance, x = summary_fun(x), shape (batch_size, ...)
        inverse          : bool, default: False
            Flag indicating whether to run the block forward or backwards
        
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
        If ``inv
        """
        
        if not inverse:
            return self.forward(target_or_z, condition)
        return self.inverse(target_or_z, condition)

    def forward(self, target, condition):
        """Performs a forward pass through a coupling layer with an optinal permutation and act norm layer."""

        # Initialize log_det_Js accumulator
        log_det_Js = tf.zeros(1)
        
        # Normalize activation, if specified
        if self.act_norm is not None:
            target, log_det_J_act = self.act_norm(target)
            log_det_Js += log_det_J_act

        # Permute, if indicated
        if self.permutation is not None:
            target = self.permutation(target)

        # Pass through coupling layer
        z, log_det_J_c = self._forward(target, condition)
        log_det_Js += log_det_J_c

        return z, log_det_Js

    def inverse(self, z, condition):
        """Performs an inverse pass through a coupling layer with an optinal permutation and act norm layer."""

        # Pass through coupling layer
        target = self._inverse(z, condition)

        # Pass through optional permutation
        if self.permutation is not None:
            target = self.permutation(target, inverse=True)
        
        # Pass through activation normalization
        if self.act_norm is not None:
            target = self.act_norm(target, inverse=True)
        return target


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
        Currently supports Gaussiand and Student-t latent spaces only.
        """
        super(InvertibleNetwork, self).__init__()

        meta = build_meta_dict(user_dict=meta,
                               default_setting=default_settings.DEFAULT_SETTING_INVERTIBLE_NET)
        self.coupling_layers = [ConditionalCouplingLayer(meta) for _ in range(meta['n_coupling_layers'])]
        if meta['adaptive_tails']:
            self.tail_network = TailNetwork(meta['tail_network'])
        else:
            self.tail_network = None
        self.z_dim = meta['n_params']

    def call(self, target, condition, inverse=False):
        """Performs one pass through an invertible chain (either inverse or forward).

        Parameters
        ----------
        target    : tf.Tensor
            The estimation quantities of interest, shape (batch_size, ...)
        condition : tf.Tensor
            The conditional data x, shape (batch_size, summary_dim)
        inverse   : bool, default: False
            Flag indicating whether to run the chain forward or backwards

        Returns
        -------
        (z, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            If inverse=False: The transformed input and the corresponding Jacobian of the transformation,
            v shape: (batch_size, ...), log_det_J shape: (batch_size, ...)

        target          :  tf.Tensor
            If inverse=True: The transformed out, shape (batch_size, ...)

        Important
        ---------
        If ``inverse=False``, the return is ``(z, log_det_J)``.\n
        If ``inverse=True``, the return is ``target``.
        """
        
        if inverse:
            return self.inverse(target, condition)
        return self.forward(target, condition)

    def forward(self, target, condition):
        """Performs a forward pass though the chain."""

        z = target
        log_det_Js = []
        for layer in self.coupling_layers:
            z, log_det_J = layer(z, condition)
            log_det_Js.append(log_det_J)
        # Sum Jacobian determinants for all layers (coupling blocks) to obtain total Jacobian.
        log_det_J = tf.add_n(log_det_Js)
        
        # Adaptive tails or simply Gaussian
        if self.tail_network is not None:
            v = self.tail_network(condition)
            return v, z, log_det_J
        else:
            return z, log_det_J

    def inverse(self, z, condition):
        """Performs a reverse pass through the chain."""

        target = z
        for layer in reversed(self.coupling_layers):
            target = layer(target, condition, inverse=True)
        return target

    def sample(self, condition, n_samples, to_numpy=True):
        """
        Samples from the inverse model given a single data instance or a batch of data instances.

        Parameters
        ----------
        condition : tf.Tensor
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
        if int(condition.shape[0]) == 1:
            # Sample from a unit Gaussian
            if self.tail_network is None:
                z_samples = tf.random.normal(shape=(n_samples, self.z_dim))
            # Sample from a t-distro
            else:
                df = self.tail_network(condition).numpy().item()
                loc = np.zeros(self.z_dim)
                shape = np.eye(self.z_dim)
                z_samples = multivariate_t(df=df, loc=loc, shape=shape).rvs(n_samples)
                
            param_samples = self.inverse(z_samples, tf.tile(condition, [n_samples, 1]))
            
        # In case of a batch input, send a 3D tensor through the invertible chain and use tensordot
        # Warning: This tensor could get pretty big if sampling a lot of values for a lot of batch instances!
        else:
            # Sample from a unit Gaussian
            if self.tail_network is None:
                z_samples = tf.random.normal(shape=(int(condition.shape[0]), n_samples, self.z_dim))
            # Sample from a t-distro    
            else:
                dfs = self.tail_network(condition).numpy().squeeze()
                loc = np.zeros(self.z_dim)
                shape = np.eye(self.z_dim)
                z_samples = tf.stack(
                    [multivariate_t(df=df, loc=loc, shape=shape).rvs(n_samples) 
                    for df in dfs]
                )
            param_samples = self.inverse(z_samples, condition)
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
        pm_samples = np.stack([np.random.dirichlet(alpha[n, :], size=n_samples) for n in range(n_datasets)], axis=0)

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

class FlattenNetwork(tf.keras.Model):
    """Implements a flattening identity mapping of inputs to outputs."""
    
    def __init__(self):
        super(FlattenNetwork, self).__init__()
    
    def call(self, x: tf.Tensor):
        """Flattens a 3D Tensor input into a 2D-tensor.
        
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, N, x_dim)
        
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, N * x_dim)
        """
        return tf.reshape(x, (x.shape[0], -1))
