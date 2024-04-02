
import keras


class Amortizer(keras.Model):
    INFERRED_VARIABLE: str
    OBSERVED_VARIABLE: str

    SUMMARY_CONDITIONS: str = "summary_conditions"
    INFERENCE_CONDITIONS: str = "inference_conditions"

    SUMMARY_TARGETS: str = "summary_targets"
    INFERENCE_TARGETS: str = "inference_targets"

    def __init__(self, inference_network, summary_network=None):
        super().__init__()
        self.inference_network = inference_network
        self.summary_network = summary_network

    def build(self, input_shape):
        if self.summary_network is not None:
            self.summary_network.build(input_shape)

        self.inference_network.build(input_shape)

    def call(self, x: dict):
        if self.summary_network:
            observed_variable, summary_conditions = x[self.OBSERVED_VARIABLE], x.get(self.SUMMARY_CONDITIONS)
            summary_outputs = self.summary_network(observed_variable, summary_conditions)
        else:
            summary_outputs = None

        inferred_variable, inference_conditions = x[self.INFERRED_VARIABLE], x.get(self.INFERENCE_CONDITIONS)
        inference_outputs = self.inference_network(inferred_variable, inference_conditions, summary_outputs)

        return {
            "inference_outputs": inference_outputs,
            "summary_outputs": summary_outputs,
        }

    def compute_loss(self, x: dict = None, y: dict = None, y_pred: dict = None, **kwargs):
        if self.summary_network:
            observed_variable, summary_conditions = x[self.OBSERVED_VARIABLE], x.get(self.SUMMARY_CONDITIONS)
            summary_loss = self.summary_network.compute_loss(
                x=(observed_variable, summary_conditions),
                y=y.get(self.SUMMARY_TARGETS),
                y_pred=y_pred["summary_outputs"]
            )
        else:
            summary_loss = keras.ops.zeros(())

        inferred_variable, inference_conditions = x[self.INFERRED_VARIABLE], x.get(self.INFERENCE_CONDITIONS)
        summaries = y_pred.get("summary_outputs")
        inference_loss = self.inference_network.compute_loss(
            x=(inferred_variable, inference_conditions, summaries),
            y=y.get(self.INFERENCE_TARGETS),
            y_pred=y_pred["inference_outputs"]
        )

        return inference_loss + summary_loss

    def compute_metrics(self, x: dict, y: dict, y_pred: dict, **kwargs):
        if self.summary_network:
            observables, conditions = x[self.OBSERVED_VARIABLE], x.get(self.SUMMARY_CONDITIONS)
            summary_metrics = self.summary_network.compute_metrics(
                x=(observables, conditions),
                y=y.get(self.SUMMARY_TARGETS),
                y_pred=y_pred["summary_outputs"]
            )
        else:
            summary_metrics = {}

        parameters, conditions = x[self.INFERRED_VARIABLE], x.get(self.INFERENCE_CONDITIONS)
        summaries = y_pred.get("summary_outputs")
        inference_metrics = self.inference_network.compute_metrics(
            x=(parameters, conditions, summaries),
            y=y.get(self.INFERENCE_TARGETS),
            y_pred=y_pred["inference_outputs"]
        )

        summary_metrics = {f"summary/{key}": value for key, value in summary_metrics.items()}
        inference_metrics = {f"inference/{key}": value for key, value in inference_metrics.items()}

        return inference_metrics | summary_metrics

    def sample(self, *args, **kwargs):
        return self.inference_network.sample(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        return self.inference_network.log_prob(*args, **kwargs)
