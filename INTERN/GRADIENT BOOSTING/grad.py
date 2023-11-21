import numpy as np
from typing import Tuple


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean squared error loss function and gradient."""
    # Вычисляем ошибку для каждого объекта
    errors = y_pred - y_true

    # Вычисляем значение функции потерь (MSE)
    loss = np.mean(errors ** 2)

    # Вычисляем градиент функции потерь (усредненный по всем объектам)
    grad = errors

    return loss, grad


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean absolute error loss function and gradient."""
    # Вычисляем ошибку для каждого объекта
    errors = y_pred - y_true

    # Вычисляем значение функции потерь (MAE)
    loss = np.mean(np.abs(errors))

    # Вычисляем градиент функции потерь (усредненный по всем объектам)
    grad = np.sign(errors)

    return loss, grad
