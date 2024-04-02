import keras

from bayesflow.experimental.types import Tensor


class Configurator:
    def __call__(self, data: dict) -> dict:
        data["parameters"] = keras.ops.concatenate(data["parameters"].values(), axis=1)
        data["observables"] = keras.ops.concatenate(data["observables"].values(), axis=1)
        return data


# class Configurator:
#     def __init__(self, posterior_summary: bool = False, likelihood_summary: bool = False):
#         super().__init__()
#         self.posterior_summary = posterior_summary
#         self.likelihood_summary = likelihood_summary
#
#     def __call__(self, data: dict) -> dict:
#         """ Configure the data for processing by neural networks """
#         data = {
#             "posterior": {"inputs": self.configure_posterior_inputs(data), "outputs": None},
#             "likelihood": {"inputs": self.configure_likelihood_inputs(data), "outputs": None},
#             **data
#         }
#
#         return data
#
#     def configure_posterior_inputs(self, data: dict) -> dict:
#         inputs = {
#             "summary": self.configure_posterior_summary_inputs(data),
#             "inference": self.configure_posterior_inference_inputs(data),
#         }
#
#         return inputs
#
#     def configure_likelihood_inputs(self, data: dict) -> dict:
#         inputs = {
#             "summary": self.configure_likelihood_summary_inputs(data),
#             "inference": self.configure_likelihood_inference_inputs(data)
#         }
#
#         return inputs
#
#     def configure_posterior_summary_inputs(self, data: dict) -> Tensor:
#         if not self.posterior_summary:
#             return None
#
#         return keras.ops.concatenate(list(data["observables"].values()), axis=2)
#
#     def configure_posterior_inference_inputs(self, data: dict) -> Tensor:
#         if not self.posterior_summary:
#             return keras.ops.concatenate(list(data["observables"].values()) + list(data["contexts"].values()), axis=1)
#
#         return keras.ops.concatenate(list(data[""]))
#
#     def configure_summary_inputs(self, parameters, conditions):
#
#     def configure_summary_inputs(self, data: dict) -> dict:
#         # TODO: more diagnostics
#         if not self.use_summary:
#             return data
#
#         # verify the default will work (no higher-dimensional tensors)
#         tensors = list(data["contexts"].values()) + list(data["observables"].values())
#         if any(keras.ops.ndim(t) > 3 for t in tensors):
#             raise NotImplementedError(f"The default summary configurator cannot handle data with ndim > 3.")
#
#         # concatenate all set tensors on the data dimension
#         set_tensors = [t for t in tensors if keras.ops.ndim(t) == 3]
#         data["summary_inputs"] = keras.ops.concatenate(set_tensors, axis=2)
#
#         return data
#
#     def configure_inference_inputs(self, data: Data) -> Data:
#         # TODO: more diagnostics
#         tensors = list(data["parameters"].values()) + list(data["contexts"].values())
#
#         if self.use_summary:
#             # concatenate only non-set tensors
#             tensors = [t for t in tensors if keras.ops.ndim(t) == 2]
#
#         data["inference_inputs"] = keras.ops.concatenate(tensors, axis=1)
#
#         return data
#
#     def configure_likelihood_inputs(self, data: Data) -> Data:
#         tensors = list(data["contexts"].values()) + list(data["parameters"].values())
#
#         # TODO: allow likelihood summary?
#         data["surrogate_inputs"] = keras.ops.concatenate(tensors, axis=1)
#         return data
#
