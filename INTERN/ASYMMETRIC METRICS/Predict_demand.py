import numpy as np


def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    ans = np.random.rand(len(y_true))
    for i, items in enumerate(zip(y_true, y_pred)):
        if items[0] >= items[1]:
            ans[i] = (items[0] - items[1])
        elif items[0] < items[1]:
            ans[i] = 0.7*(items[0] - items[1])
    error = float(np.mean(np.sqrt(np.square(ans))))
    return error
