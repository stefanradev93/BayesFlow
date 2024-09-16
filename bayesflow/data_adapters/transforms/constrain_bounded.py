from collections.abc import Sequence
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
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

    def __init__(
        self,
        parameters: str | Sequence[str] | None = None,
        /,
        *,
        lower: np.ndarray = None,
        upper: np.ndarray = None,
        method: str,
    ):
        super().__init__(parameters)

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
                    self.forward = lambda x: x
                    self.inverse = lambda x: np.clip(x, lower, upper)
                case "sigmoid":
                    self.forward = lambda x: inverse_sigmoid((x - lower) / (upper - lower))
                    self.inverse = lambda x: (upper - lower) * sigmoid(x) + lower
                case str() as name:
                    raise ValueError(f"Unsupported method name for double bounded constraint: '{name}'.")
                case other:
                    raise TypeError(f"Expected a method name, got {other!r}.")
        else:
            # single bounded case
            if lower is not None:
                match method:
                    case "clip":
                        self.forward = lambda x: x
                        self.inverse = lambda x: np.clip(x, lower, np.inf)
                    case "softplus":
                        self.forward = lambda x: inverse_softplus(x - lower)
                        self.inverse = lambda x: softplus(x) + lower
                    case "exp":
                        self.forward = np.log
                        self.inverse = np.exp
                    case str() as name:
                        raise ValueError(f"Unsupported method name for single bounded constraint: '{name}'.")
                    case other:
                        raise TypeError(f"Expected a method name, got {other!r}.")
            else:
                match method:
                    case "clip":
                        self.forward = lambda x: x
                        self.inverse = lambda x: np.clip(x, -np.inf, upper)
                    case "softplus":
                        self.forward = lambda x: -inverse_softplus(-(x - upper))
                        self.inverse = lambda x: -softplus(-x) + upper
                    case "exp":
                        self.forward = lambda x: -np.log(-x + upper)
                        self.inverse = lambda x: -np.exp(-x) + upper
                    case str() as name:
                        raise ValueError(f"Unsupported method name for single bounded constraint: '{name}'.")
                    case other:
                        raise TypeError(f"Expected a method name, got {other!r}.")

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ConstrainBounded":
        return cls(
            deserialize(config["parameters"], custom_objects),
            lower=deserialize(config["lower"], custom_objects),
            upper=deserialize(config["upper"], custom_objects),
            method=deserialize(config["method"], custom_objects),
        )

    def get_config(self) -> dict:
        return {
            "parameters": serialize(self.parameters),
            "lower": serialize(self.lower),
            "upper": serialize(self.upper),
            "method": serialize(self.method),
        }
