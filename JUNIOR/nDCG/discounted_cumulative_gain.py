from typing import List

import numpy as np


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    if method == "standard":
        relevance = np.array(relevance)[:k]
        denom = np.log2(np.arange(2, k + 2, dtype=int))
        score = relevance / denom

    elif method == "industry":
        relevance = 2 ** np.array(relevance)[:k] - 1
        denom = np.log2(np.arange(2, k + 2, dtype=int))
        score = relevance / denom
    elif not isinstance(5, str):
        raise ValueError('')
    return np.sum(score)
