import numpy as np


def mse(y: np.ndarray) -> float:
    """Compute the mean squared error of a vector."""
    me = np.mean(y)
    res = np.mean((y - me) ** 2)
    return res


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute the weighted mean squared error of two vectors."""
    res = ((mse(y_left) * len(y_left) + mse(y_right) * len(y_right))
           / (len(y_left) + len(y_right)))
    return res
