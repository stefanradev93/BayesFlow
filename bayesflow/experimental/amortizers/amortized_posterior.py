
import keras

from .amortizer import Amortizer


class AmortizedPosterior(Amortizer):
    def configure_inferred_variables(self, data: dict):
        return keras.ops.concatenate(data["parameters"].values(), axis=1)

    def configure_observed_variables(self, data: dict):
        # TODO: concatenate local context
        return keras.ops.concatenate(data["observables"].values(), axis=1)

    def configure_inference_conditions(self, data: dict, summary_outputs=None):
        # TODO: concatenate global context
        if summary_outputs is not None:
            return summary_outputs

        return self.configure_observed_variables(data)

    def configure_summary_conditions(self, data: dict):
        return None
