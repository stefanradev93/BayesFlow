import keras
import logging


def _log(msg, *args, callback_fn: callable = print, **kwargs):
    if keras.backend.backend() == "jax":
        import jax

        def __log(*a, **k):
            jax.debug.callback(callback_fn, msg.format(*a, **k))

        jax.debug.callback(__log, *args, **kwargs)
    else:
        callback_fn(msg.format(*args, **kwargs))


def critical(msg, *args, **kwargs):
    _log(msg, *args, callback_fn=logging.critical, **kwargs)


def debug(msg, *args, **kwargs):
    _log(msg, *args, callback_fn=logging.debug, **kwargs)


def error(msg, *args, **kwargs):
    _log(msg, *args, callback_fn=logging.error, **kwargs)


def exception(msg, *args, **kwargs):
    _log(msg, *args, callback_fn=logging.exception, **kwargs)


def info(msg, *args, **kwargs):
    _log(msg, *args, callback_fn=logging.info, **kwargs)


def log(msg, *args, **kwargs):
    _log(msg, *args, callback_fn=logging.log, **kwargs)


def warning(msg, *args, **kwargs):
    _log(msg, *args, callback_fn=logging.warning, **kwargs)
