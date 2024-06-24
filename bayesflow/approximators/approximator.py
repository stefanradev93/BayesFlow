import keras
from keras.saving import register_keras_serializable

from bayesflow.configurators import Configurator

match keras.backend.backend():
    case "jax":
        from .jax_approximator import JAXApproximator as BaseApproximator
    case "numpy":
        from .numpy_approximator import NumpyApproximator as BaseApproximator
    case "tensorflow":
        from .tensorflow_approximator import TensorFlowApproximator as BaseApproximator
    case "torch":
        from .torch_approximator import TorchApproximator as BaseApproximator
    case other:
        raise NotImplementedError(f"BayesFlow does not currently support backend '{other}'.")


@register_keras_serializable(package="bayesflow.amortizers")
class Approximator(BaseApproximator):
    def __init__(self, **kwargs):
        """The main workhorse for learning amortized neural approximators for distributions arising
        in inverse problems and Bayesian inference (e.g., posterior distributions, likelihoods, marginal
        likelihoods).

        The complete semantics of this class allow for flexible estimation of the following distribution:

        Q(inference_variables | H(summary_variables; summary_conditions), inference_conditions),

        # TODO - math notation

        where all quantities to the right of the "given" symbol | are optional and H refers to the optional
        summary /embedding network used to compress high-dimensional data into lower-dimensional summary
        vectors. Some examples are provided below.

        Parameters
        ----------
        inference_variables: list[str]
            A list of variable names indicating the quantities to be inferred / learned by the approximator,
            e.g., model parameters when approximating the Bayesian posterior or observables when approximating
            a likelihood density.
        inference_conditions: list[str]
            A list of variable names indicating quantities that will be used to condition (i.e., inform) the
            distribution over inference variables directly, that is, without passing through the summary network.
        summary_variables: list[str]
            A list of variable names indicating quantities that will be used to condition (i.e., inform) the
            distribution over inference variables after passing through the summary network (i.e., undergoing a
            learnable transformation / dimensionality reduction). For instance, non-vector quantities (e.g.,
            sets or time-series) in posterior inference will typically qualify as summary variables. In addition,
            these quantities may involve learnable distributions on their own.
        summary_conditions: list[str]
            A list of variable names indicating quantities that will be used to condition (i.e., inform) the
            optional summary network, e.g., when the summary network accepts further conditions that do not
            conform to the semantics of summary variable (i.e., need not be embedded or their distribution
            needs not be learned).

            # TODO add citations

        Examples
        -------
        # TODO
        """
        if "configurator" not in kwargs:
            # try to set up a default configurator
            if "inference_variables" not in kwargs:
                raise ValueError("You must specify either a configurator or arguments for the default configurator.")

            inference_variables = kwargs.pop("inference_variables")
            inference_conditions = kwargs.pop("inference_conditions", None)
            summary_variables = kwargs.pop("summary_variables", None)
            summary_conditions = kwargs.pop("summary_conditions", None)

            kwargs["configurator"] = Configurator(
                inference_variables, inference_conditions, summary_variables, summary_conditions
            )

        kwargs.setdefault("summary_network", None)
        super().__init__(**kwargs)
