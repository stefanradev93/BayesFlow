
import keras

from .amortizer import Amortizer


class AmortizedPosterior(Amortizer):
    def configure_inferred_variables(self, data: dict):
        return keras.ops.concatenate(data["parameters"].values(), axis=1)

    def configure_observed_variables(self, data: dict):
        return keras.ops.concatenate(data["observables"].values(), axis=1)

    def configure_inference_conditions(self, data: dict, summary_outputs=None):
        return summary_outputs

    def configure_summary_conditions(self, data: dict):
        return None
