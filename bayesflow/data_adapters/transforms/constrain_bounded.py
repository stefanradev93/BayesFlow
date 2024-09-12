from keras.saving import register_keras_serializable as serializable
import numpy as np

from bayesflow.utils.numpy_utils import (
    inverse_sigmoid,
    inverse_softplus,
    sigmoid,
    softplus,
)

from .transform import Transform


@serializable(package="bayesflow.data_adapters")
class ConstrainBounded(Transform):
    """Constrains a parameter with a lower and/or upper bound."""

    def __init__(self, parameter_name: str, *, lower: np.ndarray = None, upper: np.ndarray = None, method: str):
        super().__init__(parameter_name)

        self.lower = lower
        self.upper = upper
        self.method = method

        if lower is None and upper is None:
            raise ValueError("At least one of 'lower' or 'upper' must be provided.")

        if lower is not None and upper is not None:
            if np.any(lower >= upper):
                raise ValueError("The lower bound must be strictly less than the upper bound.")

            # double bounded case
            match method:
                case "clip":
                    self.forward = lambda x: np.clip(x, lower, upper)
                    self.inverse = lambda x: x
                case "sigmoid":
                    self.forward = lambda x: (upper - lower) * sigmoid(x) + lower
                    self.inverse = lambda x: inverse_sigmoid((x - lower) / (upper - lower))
                case str() as name:
                    raise ValueError(f"Unsupported method name for double bounded constraint: '{name}'.")
                case other:
                    raise TypeError(f"Expected a method name, got {other!r}.")
        else:
            # single bounded case
            if lower is not None:
                match method:
                    case "clip":
                        self.forward = lambda x: np.clip(x, lower, np.inf)
                        self.inverse = lambda x: x
                    case "softplus":
                        self.forward = lambda x: softplus(x) + lower
                        self.inverse = lambda x: inverse_softplus(x - lower)
                    case "exp":
                        self.forward = np.exp
                        self.inverse = np.log
                    case str() as name:
                        raise ValueError(f"Unsupported method name for single bounded constraint: '{name}'.")
                    case other:
                        raise TypeError(f"Expected a method name, got {other!r}.")
            else:
                match method:
                    case "clip":
                        self.forward = lambda x: np.clip(x, -np.inf, upper)
                        self.inverse = lambda x: x
                    case "softplus":
                        self.forward = lambda x: -softplus(-x) + upper
                        self.inverse = lambda x: -inverse_softplus(-(x - upper))
                    case "exp":
                        self.forward = lambda x: -np.exp(-x) + upper
                        self.inverse = lambda x: -np.log(-x + upper)
                    case str() as name:
                        raise ValueError(f"Unsupported method name for single bounded constraint: '{name}'.")
                    case other:
                        raise TypeError(f"Expected a method name, got {other!r}.")

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ConstrainBounded":
        return cls(config["parameter_name"], lower=config["lower"], upper=config["upper"], method=config["method"])

    def get_config(self) -> dict:
        return {
            "parameter_name": self.parameter_name,
            "lower": self.lower,
            "upper": self.upper,
            "method": self.method,
        }
