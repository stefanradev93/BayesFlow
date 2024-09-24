import keras

from bayesflow.types import Tensor
from bayesflow.utils import issue_url

# hard coded from keras.ops.logspace(-6, 6, 11)
# to avoid pytorch errors/warnings if you want to use MPS
default_scales = keras.ops.convert_to_tensor(
    [
        1.0000e-06,
        1.5849e-05,
        2.5119e-04,
        3.9811e-03,
        6.3096e-02,
        1.0000e00,
        1.5849e01,
        2.5119e02,
        3.9811e03,
        6.3096e04,
        1.0000e06,
    ]
)


def gaussian_kernel(x1: Tensor, x2: Tensor, scales: Tensor = default_scales) -> Tensor:
    residuals = x1[:, None] - x2[None, :]
    residuals = keras.ops.reshape(residuals, keras.ops.shape(residuals)[:2] + (-1,))
    norms = keras.ops.norm(residuals, ord=2, axis=2)
    exponent = norms[:, :, None] / (2.0 * scales[None, None, :])
    return keras.ops.mean(keras.ops.exp(-exponent), axis=2)


def maximum_mean_discrepancy(x1: Tensor, x2: Tensor, kernel: str = "gaussian", **kwargs) -> Tensor:
    """Computes the maximum mean discrepancy between samples x1 and x2.

    :param x1: Tensor of shape (n, ...)

    :param x2: Tensor of shape (n, ...)

    :param kernel: Name of the kernel to use.
        Default: 'gaussian'

    :param kwargs: Additional keyword arguments to pass to the kernel function.

    :return: Tensor of shape (n,)
        The (x1)-sample-wise maximum mean discrepancy between samples in x1 and x2.
    """
    if kernel != "gaussian":
        raise ValueError(
            "For now, we only support the Gaussian kernel. "
            f"If you need a different kernel, please open an issue at {issue_url}"
        )
    else:
        kernel_fn = gaussian_kernel

    # cannot check first (batch) dimension since it will be unknown at compile time
    if keras.ops.shape(x1)[1:] != keras.ops.shape(x2)[1:]:
        raise ValueError(
            f"Expected x1 and x2 to live in the same feature space, "
            f"but got {keras.ops.shape(x1)[1:]} != {keras.ops.shape(x2)[1:]}."
        )

    # use flattened versions
    x1 = keras.ops.reshape(x1, (keras.ops.shape(x1)[0], -1))
    x2 = keras.ops.reshape(x2, (keras.ops.shape(x2)[0], -1))

    k1 = keras.ops.mean(kernel_fn(x1, x1, **kwargs), axis=1)
    k2 = keras.ops.mean(kernel_fn(x2, x2, **kwargs), axis=1)
    k3 = keras.ops.mean(kernel_fn(x1, x2, **kwargs), axis=1)

    return k1 + k2 - 2.0 * k3
