import logging
import keras


def find_maximum_batch_size(gen_fn: callable, start: int = 1, stop: int = 2**31, error_type: type = "auto") -> int:
    """
    Find the maximum batch size for a given generator function, such that memory is filled completely.
    Note: this does not factor in autograd memory usage, so treat the estimate as an upper bound.

    :param gen_fn: A generator function that takes a batch size as input and returns a batch of samples.

    :param start: The initial batch size to start the search from.

    :param stop: The maximum batch size to search for.

    :param error_type: The type of error to catch if allocation fails.
        If "auto", will use the default error types for the backend.

    :return: The maximum batch size.
    """
    if error_type == "auto":
        match keras.backend.backend():
            case "jax":
                from jaxlib.xla_extension import XlaRuntimeError

                error_type = XlaRuntimeError
            case "numpy":
                error_type = MemoryError
            case "tensorflow":
                import tensorflow as tf

                error_type = tf.errors.ResourceExhaustedError
            case "torch":
                import torch

                # torch only throws if CUDA OOM, not CPU OOM
                error_type = torch.cuda.OutOfMemoryError

    # ensure maximum memory is available
    keras.utils.clear_session()

    batch_size = start
    while batch_size <= stop:
        try:
            # allocate memory
            _samples = gen_fn(batch_size)
        except error_type:
            # allocation did not work, return the previous batch size
            return batch_size // 2
        else:
            # allocation worked, double the batch size and try again
            batch_size *= 2
        finally:
            # always clear allocated memory, regardless if allocation worked or not
            keras.utils.clear_session()

    # reached stop condition without running out of memory
    return stop


def warning(msg, *args, **kwargs):
    match keras.backend.backend():
        case "jax":
            import jax

            def _log(*args, **kwargs):
                logging.warning(msg.format(*args, **kwargs))

            jax.debug.callback(_log, *args, **kwargs)
        case "tensorflow":
            import tensorflow as tf

            if any([tf.is_symbolic_tensor(x) for x in args]) or any(
                [tf.is_symbolic_tensor(x) for x in kwargs.values()]
            ):
                return

            logging.warning(msg.format(*args, **kwargs))
        case "numpy" | "torch":
            logging.warning(msg.format(*args, **kwargs))
