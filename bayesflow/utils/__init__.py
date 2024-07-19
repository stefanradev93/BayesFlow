from .dict_utils import (
    concatenate_dicts,
    convert_args,
    convert_kwargs,
    filter_concatenate,
    filter_kwargs,
    keras_kwargs,
    stack_dicts,
    process_output,
)

from .functional import batched_call

from .git import (
    issue_url,
    pull_url,
    repo_url,
)

from .io import (
    find_maximum_batch_size,
)

from . import logging

from .jacobian_trace import jacobian_trace

from .dispatch import find_distribution, find_network, find_permutation, find_pooling, find_recurrent_net

from .optimal_transport import optimal_transport

from .tensor_utils import (
    broadcast_right,
    broadcast_right_as,
    expand_right,
    expand_right_as,
    expand_right_to,
    expand_tile,
    size_of,
    tile_axis,
)
