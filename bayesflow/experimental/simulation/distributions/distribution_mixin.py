
from typing import Protocol


class SampleMixin(Protocol):
    def sample(self, *args, **kwargs):
        raise NotImplementedError


class LogProbMixin(Protocol):
    def log_prob(self, *args, **kwargs):
        raise NotImplementedError


class DistributionMixin(Protocol, SampleMixin, LogProbMixin):
    is_conditional: bool
