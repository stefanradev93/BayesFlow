import numpy as np
from scipy.special import softmax as scipy_softmax


def one_hot(indices: np.ndarray, num_classes: int, dtype: str = "float32") -> np.ndarray:
    """Converts a 1D array of indices to a one-hot encoded 2D array."""
    return np.eye(num_classes, dtype=dtype)[indices]


softmax = scipy_softmax
