
import inspect

from bayesflow.experimental.types import Shape, Distribution

from .joint_distribution import JointDistribution


class GenerativeModel(JointDistribution):
    """ Generate Observables Unconditionally: x ~ p(x) = ∫∫ p(x|θ,c) p(θ|c) p(c) dθ dc """
    # TODO: move to informed joint distribution
    def __init__(self, global_context=None, local_context=None, prior=None, likelihood=None):
        self.global_context = global_context
        self.local_context = local_context
        self.prior = prior
        self.likelihood = likelihood

    def sample(self, batch_shape: Shape) -> dict:
        if self.global_context is None:
            global_context = {}
        else:
            global_context = self.global_context.sample(batch_shape)

        if self.local_context is None:
            local_context = {}
        else:
            candidates = global_context
            args = self._get_args(candidates, self.local_context)
            local_context = self.local_context.sample(batch_shape, *args)

        candidates = global_context | local_context
        args = self._get_args(candidates, self.prior)
        parameters = self.prior.sample(batch_shape, *args)

        candidates = global_context | local_context | parameters
        args = self._get_args(candidates, self.likelihood)
        observables = self.likelihood.sample(batch_shape, *args)

        return dict(
            global_context=global_context,
            local_context=local_context,
            parameters=parameters,
            observables=observables,
        )

    def _get_args(self, candidates: dict, distribution: Distribution) -> list:
        if hasattr(distribution.__class__, "_raw_sample_fn"):
            signature = inspect.signature(distribution.__class__._raw_sample_fn)
        else:
            signature = inspect.signature(distribution.sample)

        args = []
        for key in signature.parameters:
            args.append(candidates[key])
        return args
