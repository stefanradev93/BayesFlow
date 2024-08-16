from .dict_utils import (
    convert_args,
    convert_kwargs,
    filter_kwargs,
    keras_kwargs,
    split_tensors,
)

from .functional import batched_call

from .git import (
    issue_url,
    pull_url,
    repo_url,
)

from .hparam_utils import find_batch_size, find_memory_budget

from .io import (
    format_bytes,
    parse_bytes,
)

from . import logging

from .jacobian_trace import jacobian_trace

from .dispatch import find_distribution, find_network, find_permutation, find_pooling, find_recurrent_net

from . import numpy_utils

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
    tree_concatenate,
    tree_stack,
)
