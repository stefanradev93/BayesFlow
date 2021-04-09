import tensorflow as tf


class MultiModelAmortizer(tf.keras.Model):

    def __init__(self, evidence_net, summary_net=None):
        """
        Connects an evidential network with a summary network as in the 
        BayesFlow for model comparison set-up.
        ----------

        evidence_net : tf.keras.Model -- an evidential network which processes the
                        outputs of multiple generative models (i.e., sim_data)
        summary_net   : tf.keras.Model or None -- an optional summary network
        """
        super(MultiModelAmortizer, self).__init__()

        self.evidence_net = evidence_net
        self.summary_net = summary_net

    def call(self, sim_data):
        """
        Performs a forward pass through the summary and inference network.
        ----------

        Arguments:
        sim_data  : tf.Tensor of shape (batch_size, n_obs, data_dim) -- the conditional data x
        ----------

        Returns:
        out : the outputs of evidence_net(summary_net(x)), usually model probabilities or absolute evidences
        """

        # Compute learnable summaries, if given
        if self.summary_net is not None:
            sim_data = self.summary_net(sim_data)

        # Compute output of inference net
        out = self.evidence_net(sim_data)
        return out

    def sample(self, obs_data, n_samples, **kwargs):
        """
        Performs inference on actually observed or simulated validation data.
        ----------

        Arguments:
        obs_data  : tf.Tensor of shape (n_datasets, n_obs, data_dim) -- the conditional data set(s)
        n_samples : int -- the number of posterior samples to obtain from the approximate posterior
        ----------

        Returns:
        post_samples : tf.Tensor of shape (n_samples, n_datasets, n_models) -- the sampled model indices or 
                       evidences per dataset or model
        """

        # Compute learnable summaries, if given
        if self.summary_net is not None:
            obs_data = self.summary_net(obs_data)

        post_samples = self.evidence_net.sample(obs_data, n_samples, **kwargs)
        return post_samples


class SingleModelAmortizer(tf.keras.Model):

    def __init__(self, inference_net, summary_net=None):
        """
        Connects an inference network for parameter estimation with an optional summary network 
        as in the original BayesFlow set-up.
        ----------

        inference_net : tf.keras.Model -- an (invertible) inference network which processes the
                        outputs of a generative model (i.e., params, sim_data)
        summary_net   : tf.keras.Model or None -- an optional summary network
        """
        super(SingleModelAmortizer, self).__init__()

        self.inference_net = inference_net
        self.summary_net = summary_net

    def call(self, params, sim_data):
        """
        Performs a forward pass through the summary and inference network.
        ----------

        Arguments:
        params    : tf.Tensor of shape (batch_size, n_params) -- the parameters theta ~ p(theta | x) of interest
        sim_data  : tf.Tensor of shape (batch_size, n_obs, data_dim) -- the conditional data x
        ----------

        Returns:
        out : the outputs of inference_net(theta, summary_net(x)), usually a latent variable and log(det(Jacobian)), that is
              a tuple (z, ldJ)
        """

        # Compute learnable summaries, if given
        if self.summary_net is not None:
            sim_data = self.summary_net(sim_data)

        # Compute output of inference net
        out = self.inference_net(params, sim_data)
        return out

    def sample(self, obs_data, n_samples, **kwargs):
        """
        Performs inference on actually observed or simulated validation data.
        ----------

        Arguments:
        obs_data  : tf.Tensor of shape (n_datasets, n_obs, data_dim) -- the conditional data set(s)
        n_samples : int -- the number of posterior samples to obtain from the approximate posterior
        ----------

        Returns:
        post_samples : tf.Tensor of shape (n_samples, n_datasets, n_params) -- the sampled parameters per data set
        """

        # Compute learnable summaries, if given
        if self.summary_net is not None:
            obs_data = self.summary_net(obs_data)

        post_samples = self.inference_net.sample(obs_data, n_samples, **kwargs)
        return post_samples
