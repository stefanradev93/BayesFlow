import tensorflow as tf

from bayesflow.exceptions import ConfigurationError, SummaryStatsError
from bayesflow.losses import kl_latent_space_gaussian, kl_latent_space_student, mmd_summary_space


class AmortizedPosterior(tf.keras.Model):
    """ Connects an inference network for parameter estimation with an optional summary network
    as in the original BayesFlow set-up.
    """

    def __init__(self, inference_net, summary_net=None, loss_fun=None, summary_loss_fun=None):
        """Initializes the SingleModelAmortizer

        Parameters
        ----------
        inference_net     : tf.keras.Model
            An (invertible) inference network which processes the outputs of a generative model 
        summary_net       : tf.keras.Model or None, optional, default: None
            An optional summary network
        loss_fun          : callable or None, optional, default: None
            The loss function which accepts the outputs of the amortizer. If None, the loss is inferred
            based on the `inference_net` type. 
        summary_loss_fun  : callable or None, optional, default: None
            The loss function which accepts the outputs of the summary network. If None, no loss is provided.

        Important
        ----------
        - If no `summary_net` is provided, then the output dictionary of your generative model should not contain
        any `sumamry_conditions`, i.e., `summary_conditions` should be set to None, otherwise these will be ignored.

        - If no custom `loss_fun` is provided, the loss function will either be a Kullback-Leibler (KL) divergence
        for a latent Gaussian space or a KL for a latent student-t space, depending on the existence of `tail_network`
        attribute in the inference net. If you are using a custom inference net mapping parameters to a latent student-t
        base distribution, make sure the inference net has a `tail_network` attribute.
        """

        super(AmortizedPosterior, self).__init__()

        self.inference_net = inference_net
        self.summary_net = summary_net
        self.summary_loss = self._determine_summary_loss(summary_loss_fun)
        self.inference_loss = self._determine_loss(loss_fun)

    def call(self, input_dict, return_summary=False, **kwargs):
        """ Performs a forward pass through the summary and inference network.

        Parameters
        ----------
        input_dict : dict  
            Input dictionary containing the following mandatory keys: 
            `parameters`         : the latent model parameters over which a condition density is learned
            `summary_conditions` : the conditioning variables (including data) that are first passed through a summary network
            `direct_conditions`  : the conditioning variables that the directly passed to the inference network
        return_summary : bool (default - False)
            A flag which determines whether the learnable data summaries (representations) are returned or not.

        Returns
        -------
        net_out or (net_out, summarized_conditions)
            the outputs of ``inference_net(theta, summary_net(x, c_s), c_d)``, usually a latent variable and
            log(det(Jacobian)), that is a tuple ``(z, log_det_J) or (sum_outputs, (z, log_det_J)) if 
            return_summary is set to True and a summary network is defined.`` 
        """
        
        # Concatenate conditions, if given
        summarized_cond, full_cond = self._compute_summary_condition(
            input_dict['summary_conditions'], 
            input_dict['direct_conditions'],
            **kwargs
        )

        # Compute output of inference net
        net_out = self.inference_net(input_dict['parameters'], full_cond, **kwargs)

        if not return_summary:
            return net_out
        return net_out, summarized_cond

    def sample(self, input_dict, n_samples, to_numpy=True, **kwargs):
        """ Performs inference on actually observed or simulated validation data.

        Parameters
        ----------
        input_dict  : dict  
            Input dictionary containing the following mandatory keys: 
            `summary_conditions` : the conditioning variables (including data) that are first passed through a summary network
            `direct_conditions`  : the conditioning variables that the directly passed to the inference network
        n_samples   : int
            The number of posterior samples to obtain from the approximate posterior

        Returns
        -------
        post_samples : tf.Tensor of shape (n_samples, n_datasets, n_params)
            the sampled parameters per data set
        """

        # Compute learnable summaries, if appropriate
        _, condition = self._compute_summary_condition(
            input_dict['summary_conditions'], 
            input_dict['direct_conditions'], 
            **kwargs
        )

        # Obtain random draws from the approximate posterior given conditioning variables
        post_samples = self.inference_net.sample(condition, n_samples, **kwargs)

        if to_numpy:
            return post_samples.numpy()
        return post_samples

    def log_posterior(self, input_dict, to_numpy=True, **kwargs):
        """ Calculates the approximate log-posterior of targets given conditional variables.

        Parameters
        ----------
        input_dict : dict  
            Input dictionary containing the following mandatory keys: 
            `parameters`         : the latent model parameters over which a condition density is learned
            `summary_conditions` : the conditioning variables (including data) that are first passed through a summary network
            `direct_conditions`  : the conditioning variables that the directly passed to the inference network

        Returns
        -------
        loglik     : tf.Tensor of shape (batch_size, n_obs)
            the approximate log-likelihood of each data point in each data set
        """

        # Compute learnable summaries, if appropriate
        _, conditions = self._compute_summary_condition(
            input_dict['summary_conditions'], 
            input_dict['direct_conditions'],
            **kwargs
        )

        # Compute approximate log posterior
        log_post = self.inference_net.log_density(input_dict['parameters'], conditions, **kwargs)

        if to_numpy:
            return log_post.numpy()
        return log_post

    def _compute_summary_condition(self, summary_conditions, direct_conditions, **kwargs):
        """ Determines how to concatenate the provided conditions.
        """

        # Compute learnable summaries, if given
        if self.summary_net is not None:
            sum_condition = self.summary_net(summary_conditions, **kwargs)
        else:
            sum_condition = None

        # Concatenate learnable summaries with fixed summaries
        if sum_condition is not None and direct_conditions is not None:
            full_cond = tf.concat([sum_condition, direct_conditions], axis=-1)
        elif sum_condition is not None:
            full_cond = sum_condition
        elif direct_conditions is not None:
            full_cond = direct_conditions
        else:
            raise SummaryStatsError("Could not determine conditioning inputs...")
        return sum_condition, full_cond

    def _determine_loss(self, loss_fun):
        """ Determines which loss to use if default None argument provided, otherwise return argument.
        """

        if loss_fun is None:
            try:
                if self.inference_net.tail_network is not None:
                    return kl_latent_space_student
                else:
                    return kl_latent_space_gaussian
            except Exception as _:
                raise ConfigurationError("Could not infer loss function based on inference net type. " +
                                         "Please provide a custom loss function!")
        elif callable(loss_fun):
            return loss_fun
        else:
            raise ConfigurationError("Loss function is neither default not callable. Please provide a valid loss function!")

    def _determine_summary_loss(self, loss_fun):
        """ Determines which summary loss to use if default None argument provided, otherwise return argument.
        """

        if loss_fun is None or callable(loss_fun):
            return loss_fun
        elif type(loss_fun) is str:
            if loss_fun == 'mmd':
                return mmd_summary_space
            else:
                raise NotImplementedError("For now, only 'mmd' is supported as a string argument for summary_loss_fun!")
        else:
            raise NotImplementedError("Could not infer summary_loss_fun, argument should be of type (None, callable, or str)!")

    def compute_loss(self, input_dict, **kwargs):
        """ Computes the loss of the amortized as specified by the init arguments.
        """

        if self.summary_loss is not None:
            net_out, sum_out = self(input_dict, return_summary=True, **kwargs)
            loss =  self.inference_loss(*net_out) + self.summary_loss(sum_out)
        else:
            net_out = self(input_dict, **kwargs)
            loss = self.inference_loss(*net_out)
        return loss


class AmortizedLikelihood(tf.keras.Model):
    """ An interface for a surrogate model of the simulator, or the implicit likelihood
    ``p(params | data, context).''
    """

    def __init__(self, surrogate_net, loss_fun=None):
        """Initializes an amortized emulator for the simulator (i.e., implicit likelihood model).

        Parameters
        ----------
        surrogate_net : tf.keras.Model
            An (invertible) inference network which processes the outputs of the generative model.
        loss_fun      : callable or None, optional, default: None
            The loss function which accepts the outputs of the amortizer. If None, the loss is inferred
            based on the `surrogate_net` type. 

        Important
        ----------
        - If no custom `loss_fun` is provided, the loss function will either be a Kullback-Leibler (KL) divergence
        for a latent Gaussian space or a KL for a latent student-t space, depending on the existence of `tail_network`
        attribute in the `surrogate_net`. If you are using a custom `surrogate_net` mapping parameters to a latent student-t
        base distribution, make sure the `surrogate_net` has a `tail_network` attribute.

        """

        super(AmortizedLikelihood, self).__init__()

        self.surrogate_net = surrogate_net
        self.loss = self._determine_loss(loss_fun)

    def call(self, input_dict, **kwargs):
        """ Performs a forward pass through the summary and inference network.

        Parameters
        ----------
        input_dict  : dict 
            Input dictionary containing the following mandatory keys: 
            `data`        - the observables over which a condition density is learned (i.e., the observables)
            `conditions`  - the conditioning variables that the directly passed to the inference network
        Returns
        -------
        net_out
            the outputs of ``inference_net(theta, summary_net(x, c_s), c_d)``, usually a latent variable and
            log(det(Jacobian)), that is a tuple ``(z, log_det_J)`` or (sum_outputs, (z, log_det_J)) if 
            return_summary is set to True and a summary network is defined.`` 
        """

        # Compute output of inference net
        net_out = self.surrogate_net(input_dict['data'], input_dict['conditions'], **kwargs)
        return net_out

    def sample(self, input_dict, n_samples, to_numpy=True, **kwargs):
        """ Performs inference on actually observed or simulated validation data.

        Parameters
        ----------

        input_dict   : dict  
            Input dictionary containing the following mandatory keys: 
            `conditions` - the conditioning variables that the directly passed to the inference network
        n_samples    : int
            The number of posterior samples to obtain from the approximate posterior
        to_numpy  : bool, default: True
            Flag indicating whether to return the samples as a `np.array` or a `tf.Tensor`
        Returns
        -------
        post_samples : tf.Tensor of shape (n_samples, n_datasets, n_params)
            the sampled parameters per data set
        """

        # Obtain random draws from the approximate posterior given conditioning variables
        post_samples = self.surrogate_net.sample(input_dict['conditions'], n_samples, **kwargs)
        if to_numpy:
            return post_samples.numpy()
        return post_samples

    def log_likelihood(self, input_dict, to_numpy=True, **kwargs):
        """ Calculates the approximate log-likelihood of targets given conditional variables.

        Parameters
        ----------
        input_dict : dict  
            Input dictionary containing the following mandatory keys: 
            `data`        - the variables over which a condition density is learned (i.e., the observables)
            `conditions`  - the conditioning variables that the directly passed to the inference network
        to_numpy   : bool, default: True
            Flag indicating whether to return the samples as a `np.array` or a `tf.Tensor`
        Returns
        -------
        log_lik     : tf.Tensor of shape (batch_size, n_obs)
            the approximate log-likelihood of each data point in each data set
        """

        log_lik = self.surrogate_net.log_density(input_dict['data'], input_dict['conditions'], **kwargs)
        if to_numpy:
            return log_lik.numpy()
        return log_lik

    def _determine_loss(self, loss_fun):
        """ Determines which loss to use if None given, otherwise return argument.
        """

        if loss_fun is None:
            try:
                if self.surrogate_net.tail_network is not None:
                    return kl_latent_space_student
                else:
                    return kl_latent_space_gaussian
            except Exception as _:
                raise ConfigurationError("Could not infer loss function based on surrogate_net type. Please input a loss function!")
        elif callable(loss_fun):
            return loss_fun
        else:
            raise ConfigurationError("Loss function is neither default not callable. Please provide a valid loss function!")

    def compute_loss(self, input_dict, **kwargs):
        """ Computes the loss of the amortized given input data provided in input_dict.

        Parameters
        ----------
        input_dict  : dict 
            Input dictionary containing the following mandatory keys: 
            `data`        - the observables over which a condition density is learned (i.e., the observables)
            `conditions`  - the conditioning variables that the directly passed to the inference network
        """

        net_out = self(input_dict, **kwargs)
        loss =  self.loss(*net_out)
        return loss
            

class JointAmortizer(tf.keras.Model):
    """ An interface for jointly learning a surrogate model of the simulator and an approximate
    posterior given a generative model.
    """

    def __init__(self, amortized_posterior, amortized_likelihood):
        """Initializes a joint learner comprising an amortized posterior and an amortized emulator.

        Parameters
        ----------
        amortized_posterior  : an instance of AmortizedPosterior or tf.keras.Model
            The generative neural posterior approximator.
        amortized_likelihood : an instance of AmortizedEmulator or tf.keras.Model
            The generative neural likelihood approximator.

        Important
        ----------

        """

        super(JointAmortizer, self).__init__()

        self.amortized_posterior = amortized_posterior
        self.amortized_likelihood = amortized_likelihood

    def call(self, input_dict, **kwargs):
        """ Performs a forward pass through both networks.
        """

        post_out = self.amortized_posterior(input_dict, **kwargs)
        lik_out = self.amortized_likelihood(input_dict, **kwargs)
        return post_out, lik_out

    def compute_loss(self, input_dict, **kwargs):
        """
        TODO
        """

        loss_post = self.amortiozed_posterior.compute_loss(input_dict, **kwargs)
        loss_lik = self.amortiozed_likelihood.compute_loss(input_dict, **kwargs)
        total_loss = loss_post + loss_lik
        return total_loss

    def log_likelihood(self, input_dict, to_numpy=True, **kwargs):
        """ Calculates the approximate log-likelihood of data given conditional variables.

        Parameters
        ----------
        input_dict : dict  
            Input dictionary containing the following mandatory keys: 
            `data`        - the variables over which a condition density is learned (i.e., the observables)
            `conditions`  - the conditioning variables that the directly passed to the inference network
        to_numpy   : bool, default: True
            Flag indicating whether to return the samples as a `np.array` or a `tf.Tensor`
        Returns
        -------
        log_lik     : tf.Tensor of shape (batch_size, n_obs)
            the approximate log-likelihood of each data point in each data set
        """

        return self.amortized_likelihood.log_likelihood(input_dict, to_numpy=to_numpy, **kwargs)
   
    def log_posterior(self, input_dict, to_numpy=True, **kwargs):
        """ Calculates the approximate log-posterior of targets given conditional variables.

        Parameters
        ----------
        input_dict : dict  
            Input dictionary containing the following mandatory keys: 
            `parameters`         : the latent model parameters over which a condition density is learned
            `summary_conditions` : the conditioning variables (including data) that are first passed through a summary network
            `direct_conditions`  : the conditioning variables that the directly passed to the inference network

        Returns
        -------
        log_post    : tf.Tensor of shape (batch_size, n_obs)
            the approximate log-likelihood of each data point in each data set
        """

        return self.amortized_posterior.log_posterior(input_dict, to_numpy=to_numpy, **kwargs)
   
    def sample_data(self, input_dict, n_samples, to_numpy=True, **kwargs):
        
        return self.amortized_likelihood.sample(input_dict, n_samples, to_numpy=to_numpy, **kwargs)

    def sample_parameters(self, input_dict, n_samples, to_numpy=True, **kwargs):
        
        return self.amortized_posterior.sample(input_dict, n_samples, to_numpy=to_numpy, **kwargs)