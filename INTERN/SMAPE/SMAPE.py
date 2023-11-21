import numpy as np


def smape(y_true: np.array, y_pred: np.array) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator = denominator.astype(float)
    denominator[denominator == 0] = np.inf
    return np.mean(2 * np.abs(y_true - y_pred) / denominator)
