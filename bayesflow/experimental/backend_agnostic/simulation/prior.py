
import functools
import inspect
import warnings


from .parameters import SampleParametersMixin


def prior(sample_fn: callable):
    target_signature = inspect.signature(SampleParametersMixin.sample_parameters)
    try:
        given_signature = inspect.signature(sample_fn)
    except ValueError | TypeError:
        # no signature can be provided, or type not supported
        warnings.warn(f"Could not validate signature for prior {sample_fn.__name__}.", UserWarning)
    else:
        # validate signature
        given_parameters = list(given_signature.parameters.values())
        target_parameters = list(target_signature.parameters.values())[1:]

        for (given_param, target_param) in zip(given_parameters, target_parameters):
            # TODO: nicer errors, e.g. using sets
            # TODO: should we validate the names? typically, args are passed positionally
            if given_param.name != target_param.name:
                raise ValueError(f"Name mismatch.")
            if given_param.default != target_param.default:
                raise ValueError(f"Default value mismatch.")

    class Prior(SampleParametersMixin):
        def sample_parameters(self, batch_shape, contexts=None):
            return sample_fn(batch_shape, contexts)

        def __call__(self, *args, **kwargs):
            return self.sample_parameters(*args, **kwargs)

    instance = Prior()

    # assign metadata
    functools.update_wrapper(instance, sample_fn)

    return instance
