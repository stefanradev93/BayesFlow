from .dictutils import (
    batched_call,
    concatenate_dicts,
    filter_concatenate,
    filter_kwargs,
    keras_kwargs,
    stack_dicts,
)

from .git import (
    issue_url,
    pull_url,
    repo_url,
)

from .jacobian_trace import jacobian_trace

from .dispatch import find_distribution, find_network, find_permutation, find_pooling, find_recurrent_net

from .optimal_transport import optimal_transport
