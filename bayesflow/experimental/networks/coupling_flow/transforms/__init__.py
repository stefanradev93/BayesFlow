
from .affine_transform import AffineTransform
from .transform import Transform


def find_transform(transform: str | Transform | type(Transform)) -> Transform:
    if isinstance(transform, Transform):
        return transform
    if isinstance(transform, type):
        return transform()

    match transform:
        case "affine":
            return AffineTransform()
        case str() as unknown_transform:
            raise ValueError(f"Unknown transform: '{unknown_transform}'")
        case other:
            raise TypeError(f"Unknown transform type: {other}")
