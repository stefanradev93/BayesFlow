
import keras
import numpy as np

from .transform import Transform


class AffineTransform(Transform):
    def split_parameters(self, parameters):
        scale, intercept = keras.ops.split(parameters, 2, axis=1)

        return {"scale": scale, "intercept": intercept}

    def constrain_parameters(self, parameters):
        shift = np.log(np.e - 1)
        parameters["scale"] = keras.ops.softplus(parameters["scale"] + shift)

        return parameters

    def forward_jacobian(self, x, parameters):
        logdet = keras.ops.sum(keras.ops.log(parameters["scale"]))
        return parameters["scale"] * x + parameters["intercept"], logdet

    def inverse_jacobian(self, z, parameters):
        logdet = -utils.sum_except
        return (z - parameters["intercept"]) / parameters["scale"], logdet
