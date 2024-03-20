
class SampleMixin:
    def sample(self, *args, **kwargs):
        raise NotImplementedError


class LogProbMixin:
    def log_prob(self, *args, **kwargs):
        raise NotImplementedError


class DistributionMixin(SampleMixin, LogProbMixin):
    pass
