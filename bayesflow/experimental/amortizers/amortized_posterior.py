
import keras

from .amortizer import Amortizer


class AmortizedPosterior(Amortizer):
    def configure_inferred_variables(self, data: dict):
        return keras.ops.concatenate(list(data["parameters"].values()), axis=1)

    def configure_observed_variables(self, data: dict):
        return keras.ops.concatenate(list(data["observables"].values()), axis=1)

    def configure_inference_conditions(self, data: dict, summary_outputs=None):
        if summary_outputs is None:
            return self.configure_observed_variables(data)

        return summary_outputs

    def configure_summary_conditions(self, data: dict):
        return None
