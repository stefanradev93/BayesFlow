import math

from keras import ops
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from .transform import Transform


@serializable(package="bayesflow.networks.coupling_flow")
class SplineTransform(Transform):
    def __init__(self, bins=16, default_domain=(-5.0, 5.0, -5.0, 5.0), **kwargs):
        super().__init__(**kwargs)

        self.bins = bins
        self.default_domain = default_domain
        self.spline_params_counts = {
            "left_edge": 1,
            "bottom_edge": 1,
            "widths": self.bins,
            "heights": self.bins,
            "derivatives": self.bins - 1,
        }
        self.split_idx = ops.cumsum(list(self.spline_params_counts.values()))[:-1]
        self._params_per_dim = sum(self.spline_params_counts.values())

        # Pre-compute defaults and softplus shifts
        default_width = (self.default_domain[1] - self.default_domain[0]) / self.bins
        default_height = (self.default_domain[3] - self.default_domain[2]) / self.bins
        self.xshift = math.log(math.exp(default_width) - 1)
        self.yshift = math.log(math.exp(default_height) - 1)
        self.softplus_shift = math.log(math.e - 1.0)

    def split_parameters(self, parameters: Tensor) -> dict[str, Tensor]:
        # Ensure spline works for 2D (batch_size, dim) and 3D (batch_size, num_reps, dim)
        shape = ops.shape(parameters)
        rank = len(shape)
        if rank == 2:
            new_shape = (shape[0], -1, self._params_per_dim)
        elif rank == 3:
            new_shape = (shape[0], shape[1], -1, self._params_per_dim)
        else:
            raise NotImplementedError("Spline flows can currently only operate on 2D and 3D inputs!")

        # Arrange spline parameters into a dictionary
        parameters = ops.reshape(parameters, new_shape)
        parameters = ops.split(parameters, self.split_idx, axis=-1)
        parameters = dict(
            left_edge=parameters[0],
            bottom_edge=parameters[1],
            widths=parameters[2],
            heights=parameters[3],
            derivatives=parameters[4],
        )
        return parameters

    @property
    def params_per_dim(self):
        return self._params_per_dim

    def constrain_parameters(self, parameters: dict[str, Tensor]) -> dict[str, Tensor]:
        # Set lower corners of domain relative to default domain
        parameters["left_edge"] = parameters["left_edge"] + self.default_domain[0]
        parameters["bottom_edge"] = parameters["bottom_edge"] + self.default_domain[2]

        # Constrain widths and heights to be positive
        parameters["widths"] = ops.softplus(parameters["widths"] + self.xshift)
        parameters["heights"] = ops.softplus(parameters["heights"] + self.yshift)

        # Compute spline derivatives
        parameters["derivatives"] = ops.softplus(parameters["derivatives"] + self.softplus_shift)

        # Add in edge derivatives
        total_width = ops.sum(parameters["widths"], axis=-1, keepdims=True)
        total_height = ops.sum(parameters["heights"], axis=-1, keepdims=True)
        scale = total_height / total_width
        parameters["derivatives"] = ops.concatenate([scale, parameters["derivatives"], scale], axis=-1)
        return parameters

    def forward(self, x: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        raise NotImplementedError

    def inverse(self, z: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        raise NotImplementedError
