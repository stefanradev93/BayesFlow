from collections.abc import Sequence
import keras
import numpy as np
from typing import TypeVar

from bayesflow.types import Shape, Tensor


T = TypeVar("T")


def broadcast_right(x: Tensor, shape: Shape) -> Tensor:
    """Broadcast x to the given shape, expanding to the right as necessary"""
    return keras.ops.broadcast_to(expand_right_to(x, len(shape)), shape)


def broadcast_right_as(x: Tensor, y: Tensor) -> Tensor:
    """Broadcast x to the shape of y, expanding to the right as necessary"""
    return broadcast_right(x, keras.ops.shape(y))


def expand_right(x: Tensor, n: int) -> Tensor:
    """Expand x to the right n times"""
    if n < 0:
        raise ValueError(f"Cannot expand {n} times.")

    idx = [...] + [None] * n
    return x[tuple(idx)]


def expand_right_to(x: Tensor, dim: int) -> Tensor:
    """Expand x to the right, matching dim"""
    return expand_right(x, dim - keras.ops.ndim(x))


def expand_right_as(x: Tensor, y: Tensor) -> Tensor:
    """Expand x to the right, matching the dimension of y"""
    return expand_right_to(x, keras.ops.ndim(y))


def expand_tile(x: Tensor, axis: int, n: int) -> Tensor:
    """Expand and tile x along the given axis n times"""
    x = keras.ops.expand_dims(x, axis=axis)
    return tile_axis(x, axis, n)


def size_of(x) -> int:
    """
    :param x: A nested structure of tensors.
    :return: The total memory footprint of x, ignoring view semantics, in bytes.
    """
    if keras.ops.is_tensor(x) or isinstance(x, np.ndarray):
        return int(keras.ops.size(x)) * np.dtype(keras.ops.dtype(x)).itemsize

    # flatten nested structure
    x = keras.tree.flatten(x)

    # get unique tensors by id
    x = {id(tensor): tensor for tensor in x}

    # sum up individual sizes
    return sum(size_of(tensor) for tensor in x.values())


def tile_axis(x: Tensor, axis: int, n: int) -> Tensor:
    """Tile x along the given axis n times"""
    repeats = [1] * keras.ops.ndim(x)
    repeats[axis] = n
    return keras.ops.tile(x, repeats=repeats)


# we want to annotate this as Sequence[PyTree[Tensor]], but static type checkers do not support PyTree's type expansion
def tree_concatenate(structures: Sequence[T], axis: int = 0, numpy: bool = False) -> T:
    """Concatenate all tensors in the given sequence of nested structures.
    All objects in the given sequence must have the same structure.
    The output will adhere to this structure.

    :param structures: A sequence of nested structures of tensors.
        All structures in the sequence must have the same layout.
        Tensors in the same layout location must have compatible shapes for concatenation.
    :param axis: The axis along which to concatenate tensors.
    :param numpy: If true, uses numpy for concatenation, ensuring all tensors remain on the cpu.
    :return: A structure of concatenated tensors with the same layout as each input structure.
    """
    if numpy:

        def fn(*tensors):
            return np.concatenate(tensors, axis=axis)
    else:

        def fn(*tensors):
            return keras.ops.concatenate(tensors, axis=axis)

    return keras.tree.map_structure(fn, *structures)


def tree_stack(structures: Sequence[T], axis: int = 0, numpy: bool = False) -> T:
    """Like :func:`tree_concatenate`, except tensors are stacked instead of concatenated."""
    if numpy:

        def fn(*tensors):
            return np.stack(tensors, axis=axis)
    else:

        def fn(*tensors):
            return keras.ops.stack(tensors, axis=axis)

    return keras.tree.map_structure(fn, *structures)
