import logging
import keras


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
