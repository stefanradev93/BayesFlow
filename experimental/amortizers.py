import tensorflow as tf
import numpy as np


class MetaAmortizer(tf.keras.Model):

    def __init__(self, inference_net=None, evidence_net=None, summary_net=None):
        """
        Connects an evidential network with a summary network as in the 
        BayesFlow for model comparison set-up.
        ----------

        inference_net : tf.keras.Model -- an (invertible) inference network which processes the
                        outputs of a generative model (i.e., params, sim_data)
        evidence_net  : tf.keras.Model -- an evidential network which processes the
                        outputs of multiple generative models (i.e., sim_data)
        summary_net   : tf.keras.Model or None -- an optional summary network
        """

    def call(self, model_indices, params, sim_data):
        """
        Performs a forward pass through the networks.
        ----------

        Arguments:
        model_indices  : tf.Tensor or np.array of shape (n_sim, n_models) 
                         -- the true, one-hot-encoded model indices m ~ p(m)
        params         : tf.Tensor or np.array of shape (n_sim, n_params) 
                         -- the parameters theta ~ p(theta | m) of interest
        sim_data       : tf.Tensor or np.array of shape (n_sim, n_obs, data_dim) 
                         -- the conditional data x
        ----------

        Returns:
        (out_inference, out_evidence) -- the outputs of the evidence and inference networks or None if 
                        no networks provided

        """
        

        # TODO - Model aware or model-agnostic summary
        if self.summary_net is not None:
            sim_data = self.summary_net(sim_data, model_indices)

        if self.evidence_net is not None:
            out_evidence = self.evidence_net(sim_data)
        else:
            out_evidence is None
        
        if self.inference_net is not None:
            out_inference = self.inference_net(model_indices, params, sim_data)
        else:
            out_inference = None
        return out_inference, out_evidence
    
    def sample_from_model(self, x, m_idx, n_samples, to_numpy=True):
        """
        Performs fast parallelized inference on a single model.
        """
    
        raise NotImplementedError('TODO!')   

    def compare_models(self):

        raise NotImplementedError('TODO!')   