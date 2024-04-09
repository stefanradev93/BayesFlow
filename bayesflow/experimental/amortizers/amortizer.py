
import keras


class Amortizer(keras.Model):
    def __init__(self, inference_network, summary_network=None):
        super().__init__()
        self.inference_network = inference_network
        self.summary_network = summary_network

    def build(self, input_shape):
        if self.summary_network is not None:
            self.summary_network.build(input_shape)

        self.inference_network.build(input_shape)

    def call(self, x: dict, training=False):
        inferred_variables = self.configure_inferred_variables(x)
        observed_variables = self.configure_observed_variables(x)

        if self.summary_network:
            summary_conditions = self.configure_summary_conditions(x)
            summary_outputs = self.summary_network(observed_variables, summary_conditions)
        else:
            summary_outputs = None

        inference_conditions = self.configure_inference_conditions(x, summary_outputs)
        inference_outputs = self.inference_network(inferred_variables, observed_variables, inference_conditions, summary_outputs)

        return {
            "inference_outputs": inference_outputs,
            "summary_outputs": summary_outputs,
        }

    def compute_loss(self, x: dict = None, y: dict = None, y_pred: dict = None, **kwargs):
        inferred_variables = self.configure_inferred_variables(x)
        observed_variables = self.configure_observed_variables(x)

        if self.summary_network:
            summary_conditions = self.configure_summary_conditions(x)
            summary_loss = self.summary_network.compute_loss(
                x=(observed_variables, summary_conditions),
                y=y.get("summary_targets"),
                y_pred=y_pred.get("summary_outputs")
            )
        else:
            summary_loss = keras.ops.zeros(())

        inference_conditions = self.configure_inference_conditions(x, y_pred.get("summary_outputs"))
        inference_loss = self.inference_network.compute_loss(
            x=(inferred_variables, observed_variables, inference_conditions, y_pred.get("summary_outputs")),
            y=y.get("inference_targets"),
            y_pred=y_pred.get("inference_outputs")
        )

        return inference_loss + summary_loss

    def compute_metrics(self, x: dict, y: dict, y_pred: dict, **kwargs):
        inferred_variables = self.configure_inferred_variables(x)
        observed_variables = self.configure_observed_variables(x)

        if self.summary_network:
            summary_conditions = self.configure_summary_conditions(x)
            summary_metrics = self.summary_network.compute_metrics(
                x=(observed_variables, summary_conditions),
                y=y.get("summary_targets"),
                y_pred=y_pred.get("summary_outputs")
            )
        else:
            summary_metrics = {}

        inference_conditions = self.configure_inference_conditions(x, y_pred.get("summary_outputs"))
        inference_metrics = self.inference_network.compute_metrics(
            x=(inferred_variables, observed_variables, inference_conditions, y_pred.get("summary_outputs")),
            y=y.get("inference_targets"),
            y_pred=y_pred.get("inference_outputs")
        )

        summary_metrics = {f"summary/{key}": value for key, value in summary_metrics.items()}
        inference_metrics = {f"inference/{key}": value for key, value in inference_metrics.items()}

        return inference_metrics | summary_metrics

    def sample(self, *args, **kwargs):
        return self.inference_network.sample(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        return self.inference_network.log_prob(*args, **kwargs)

    def configure_inferred_variables(self, data: dict):
        """
        Return the inferred variables, given the data.
        Inferred variables are passed as input to the inference network.

        This method must be efficient and deterministic.
        Best practice is to prepare the output in dataset.__getitem__,
        which is run in a worker process, and then simply fetch a key from the data dictionary here.
        """
        raise NotImplementedError

    def configure_observed_variables(self, data: dict):
        """
        Return the observed variables, given the data.
        Observed variables are passed as input to the summary and/or inference networks.

        This method must be efficient and deterministic.
        Best practice is to prepare the output in dataset.__getitem__,
        which is run in a worker process, and then simply fetch a key from the data dictionary here.
        """
        raise NotImplementedError

    def configure_inference_conditions(self, data: dict, summary_outputs=None):
        """
        Return the inference conditions, given the data.
        Inference conditions are passed as conditional input to the inference network.

        If summary outputs are provided, they should be concatenated to the return value.

        This method must be efficient and deterministic.
        Best practice is to prepare the output in dataset.__getitem__,
        which is run in a worker process, and then simply fetch a key from the data dictionary here.
        """
        raise NotImplementedError

    def configure_summary_conditions(self, data: dict):
        """
        Return the summary conditions, given the data.
        Summary conditions are passed as conditional input to the summary network.

        This method must be efficient and deterministic.
        Best practice is to prepare the output in dataset.__getitem__,
        which is run in a worker process, and then simply fetch a key from the data dictionary here.
        """
        raise NotImplementedError
