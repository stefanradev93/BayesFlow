
import keras
from keras.saving import (
    deserialize_keras_object,
    register_keras_serializable,
    serialize_keras_object,
)

from bayesflow.experimental.types import Tensor
from bayesflow.experimental.networks import InferenceNetwork, SummaryNetwork


class BaseApproximator(keras.Model):
    def __init__(
        self,
        inference_network: InferenceNetwork,
        summary_network: SummaryNetwork = None,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.inference_network = inference_network
        self.summary_network = summary_network

    def sample(self, num_samples: int, **kwargs) -> dict[str, Tensor]:
        # TODO
        return {}

    def log_prob(self, samples: dict[str, Tensor], **kwargs) -> Tensor:
        # TODO
        return {}

    def call(self, *, training=False, **data):
        if not training:
            # user-called
            raise NotImplementedError(
                f"{self.__class__.__name__}.call is not well-defined and thus intentionally not implemented.\n"
                f"For high-level sampling, use the `sample` and `log_prob` methods instead.\n"
                f"For a low-level forward-pass, use the internal `inference_network` and `summary_network` directly."
            )

        # possibly keras-internal called, so we cannot raise
        return None

    def compute_loss(self, **data):

        # Configure dict outputs into tensors
        inference_variables = self.configure_inference_variables(data)
        inference_conditions = self.configure_inference_conditions(data)
        summary_variables = self.configure_summary_variables(data)
        summary_conditions = self.configure_summary_conditions(data)

        # Obtain summary outputs and summary loss (if present)
        if self.summary_network:
            summary_outputs = self.summary_network(summary_variables, summary_conditions)
            summary_loss = self.summary_network.compute_loss(summary_outputs)
        else:
            summary_outputs = None
            summary_loss = keras.ops.zeros(())

        # Combine summary outputs and inference conditions
        full_conditions = self.configure_full_conditions(summary_outputs, inference_conditions)

        # Compute inference loss
        inference_loss = self.inference_network.compute_loss(
            targets=inference_variables,
            conditions=full_conditions,
        )

        return inference_loss + summary_loss

    def compute_metrics(self, **data):
        #TODO
        base_metrics = super().compute_metrics(**data)
        return base_metrics

        # inference_variables = self.configure_inference_variables(data)
        # inference_conditions = self.configure_inference_conditions(data)
        # summary_variables = self.configure_summary_variables(data)
        # summary_conditions = self.configure_summary_conditions(data)
        #
        # if self.summary_network:
        #     summary_metrics = self.summary_network.compute_metrics(
        #         summary_variables=summary_variables,
        #         summary_conditions=summary_conditions,
        #     )
        # else:
        #     summary_metrics = {}
        #
        # inference_metrics = self.inference_network.compute_metrics(
        #     inference_variables=inference_variables,
        #     conditions=conditions,
        # )
        #
        # summary_metrics = {f"summary/{key}": value for key, value in summary_metrics.items()}
        # inference_metrics = {f"inference/{key}": value for key, value in inference_metrics.items()}

        # return base_metrics | inference_metrics | summary_metrics

    def configure_full_conditions(
        self,
        summary_outputs: Tensor | None,
        inference_conditions: Tensor | None,
    ) -> Tensor:
        """
        Combine the (optional) inference conditions with the (optional) outputs
        of the (optional) summary network.
        """

        raise NotImplementedError

    def configure_inference_variables(self, data: dict) -> any:
        """
        Return the inferred variables, given the data.
        Inferred variables are passed as input to the inference network.

        This method must be efficient and deterministic.
        Best practice is to prepare the output in dataset.__getitem__,
        which is run in a worker process, and then simply fetch a key from the data dictionary here.
        """
        raise NotImplementedError

    def configure_inference_conditions(self, data: dict) -> any:
        """
        Return the inference conditions, given the data.
        Inference conditions are passed as conditional input to the inference network.

        If summary outputs are provided, they should be concatenated to the return value.

        This method must be efficient and deterministic.
        Best practice is to prepare the output in dataset.__getitem__,
        which is run in a worker process, and then simply fetch a key from the data dictionary here.
        """
        raise NotImplementedError

    def configure_summary_variables(self, data: dict) -> any:
        """
        Return the observed variables, given the data.
        Observed variables are passed as input to the summary and/or inference networks.

        This method must be efficient and deterministic.
        Best practice is to prepare the output in dataset.__getitem__,
        which is run in a worker process, and then simply fetch a key from the data dictionary here.
        """
        raise NotImplementedError

    def configure_summary_conditions(self, data: dict) -> any:
        """
        Return the summary conditions, given the data.
        Summary conditions are passed as conditional input to the summary network.

        This method must be efficient and deterministic.
        Best practice is to prepare the output in dataset.__getitem__,
        which is run in a worker process, and then simply fetch a key from the data dictionary here.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "BaseApproximator":
        inference_network = deserialize_keras_object(config.pop("inference_network"), custom_objects=custom_objects)
        summary_network = deserialize_keras_object(config.pop("summary_network"), custom_objects=custom_objects)

        return cls(inference_network, summary_network, **config)

    def get_config(self):
        base_config = super().get_config()

        config = {
            "inference_network": serialize_keras_object(self.inference_network),
            "summary_network": serialize_keras_object(self.summary_network),
        }

        return base_config | config
