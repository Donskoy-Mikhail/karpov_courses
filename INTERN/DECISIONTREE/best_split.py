from __future__ import annotations
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


def best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Find the best split for all features"""

    best_feature = None
    best_threshold = None
    best_weighted_mse = float('inf')

    for feature in range(X.shape[1]):
        threshold = split(X, y, feature)  # Используем ранее реализованную функцию split

        # Разделяем данные на две подвыборки: слева и справа от порога
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        # Вычисляем взвешенный MSE для подвыборок
        left_mse = np.mean((y[left_indices] - np.mean(y[left_indices])) ** 2)
        right_mse = np.mean((y[right_indices] - np.mean(y[right_indices])) ** 2)

        # Вычисляем взвешенный MSE для текущего признака и порога
        weighted_mse = (left_mse * np.sum(left_indices) + right_mse * np.sum(right_indices)) / len(y)

        # Если текущий признак и порог дают меньший взвешенный MSE, обновляем значения
        if weighted_mse < best_weighted_mse:
            best_weighted_mse = weighted_mse
            best_feature = feature
            best_threshold = threshold

    return best_feature, best_threshold
