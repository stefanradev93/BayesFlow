
from .distribution import Distribution
from .normal import Normal


def find_distribution(distribution: str | Distribution | type(Distribution)) -> Distribution:
    if isinstance(distribution, Distribution):
        return distribution
    if isinstance(distribution, type):
        return Distribution()

    match distribution:
        case "normal":
            distribution = Normal()
        case str() as unknown_distribution:
            raise ValueError(f"Distribution '{unknown_distribution}' is unknown or not yet supported by name.")
        case other:
            raise TypeError(f"Unknown distribution type: {other}")

    return distribution
