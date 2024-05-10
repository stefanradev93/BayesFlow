
from bayesflow.experimental.types import Shape


class JointDistribution:
    def sample(self, batch_shape: Shape) -> dict:
        raise NotImplementedError
