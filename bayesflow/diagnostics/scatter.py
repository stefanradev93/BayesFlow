import matplotlib.pyplot as plt

from bayesflow.types import Tensor


def scatter(samples: dict[str, Tensor], parameter: str, dims: (int, int) = (0, 1), **kwargs):
    x = samples[parameter][..., dims[0]]
    y = samples[parameter][..., dims[1]]

    artist = plt.scatter(x, y, **kwargs)
    plt.xlabel(rf"${parameter}_{dims[0] + 1}$")
    plt.ylabel(rf"${parameter}_{dims[1] + 1}$")

    return artist
