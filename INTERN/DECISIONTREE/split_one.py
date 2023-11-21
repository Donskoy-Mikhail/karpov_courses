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


def split(X: np.ndarray, y: np.ndarray, feature: int) -> float:
    """Find the best split for a node (one feature)"""

    unique_values = np.unique(X[:, feature])

    best_threshold = None
    best_weighted_mse = float('inf')

    for threshold in unique_values:
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        left_mse = np.mean((y[left_indices] - np.mean(y[left_indices])) ** 2)
        right_mse = np.mean((y[right_indices] - np.mean(y[right_indices])) ** 2)

        weighted_mse = (left_mse * np.sum(left_indices) + right_mse * np.sum(right_indices)) / len(y)

        if weighted_mse < best_weighted_mse:
            best_weighted_mse = weighted_mse
            best_threshold = threshold

    return best_threshold
