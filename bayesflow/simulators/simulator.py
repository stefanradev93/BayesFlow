from bayesflow.types import Shape, Tensor


class Simulator:
    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, Tensor]:
        raise NotImplementedError
