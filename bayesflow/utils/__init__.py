from .dict_utils import (
    batched_call,
    concatenate_dicts,
    filter_concatenate,
    filter_kwargs,
    keras_kwargs,
    process_output,
    stack_dicts,
)
from .dispatch import find_distribution, find_network, find_permutation, find_pooling, find_recurrent_net
from .git import (
    issue_url,
    pull_url,
    repo_url,
)
from .io import warning
from .jacobian_trace import jacobian_trace
from .optimal_transport import optimal_transport
from .tensor_utils import (
    broadcast_right,
    broadcast_right_as,
    expand_right,
    expand_right_as,
    expand_right_to,
    expand_tile,
    tile_axis,
)
