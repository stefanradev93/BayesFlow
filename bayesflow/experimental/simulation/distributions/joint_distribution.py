
import inspect
import keras
from typing import Sequence

from bayesflow.experimental.types import Distribution, Shape


class JointDistribution:
    def sample(self, batch_shape: Shape) -> dict:
        raise NotImplementedError


class SequentialFactorization(JointDistribution):
    def __init__(self, distributions: Sequence[Distribution]):
        super().__init__()
        self.distributions = distributions

    def sample(self, batch_shape: Shape) -> dict:
        samples = {}

        for distribution in self.distributions:
            args = self._get_args(samples, distribution)
            samples |= distribution.sample(batch_shape, *args)

        return samples

    def _get_args(self, candidates: dict, distribution: Distribution) -> list:
        if hasattr(distribution.__class__, "_raw_sample_fn"):
            signature = inspect.signature(distribution.__class__._raw_sample_fn)
        else:
            signature = inspect.signature(distribution.sample)

        args = []
        for key in signature.parameters:
            args.append(candidates[key])
        return args


class DefaultFactorization(SequentialFactorization):
    """ Generate Observables Unconditionally: x ~ p(x) = ∫∫ p(x|θ,c) p(θ|c) p(c) dθ dc """
    def __init__(self, global_context=None, local_context=None, prior=None, likelihood=None):
        # TODO: broadcast global context
        distributions = [global_context, local_context, prior, likelihood]
        distributions = [d for d in distributions if d is not None]
        super().__init__(distributions)
