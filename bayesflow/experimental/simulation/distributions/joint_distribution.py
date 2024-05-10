
from bayesflow.experimental.types import Shape


class JointDistribution:
    def __init__(self, prior, likelihood):
        self.prior = prior
        self.likelihood = likelihood

    def sample(self, batch_shape: Shape) -> dict:
        parameters = self.prior.sample(batch_shape)
        observables = self.likelihood.sample(batch_shape, **parameters)

        return dict(parameters=parameters, observables=observables)
