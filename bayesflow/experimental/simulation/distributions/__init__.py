
from bayesflow.experimental.types import Distribution, Shape

from .joint_distribution import JointDistribution
from .spherical_gaussian import SphericalGaussian


def find_distribution(distribution: str | Distribution | type(Distribution)) -> Distribution:
    if isinstance(distribution, Distribution):
        return distribution
    if isinstance(distribution, type):
        return Distribution()
    match distribution:
        case "normal":
            distribution = SphericalGaussian()
        case str() as unknown_distribution:
            raise ValueError(f"Distribution '{unknown_distribution}' is unknown or not yet supported by name.")
        case other:
            raise TypeError(f"Unknown distribution type: {other}")

    return distribution
