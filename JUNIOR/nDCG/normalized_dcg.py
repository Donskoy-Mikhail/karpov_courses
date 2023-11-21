from typing import List

import numpy as np


def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain.

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
        relevance_i = relevance.copy()
        relevance_i = sorted(relevance_i, reverse=True)
        relevance_i = np.array(relevance_i)[:k]
        denom_i = np.log2(np.arange(2, k + 2, dtype=int))
        score_i = relevance_i / denom_i

        relevance = np.array(relevance)[:k]
        denom = np.log2(np.arange(2, k + 2, dtype=int))
        score = np.sum((relevance / denom)) / np.sum(score_i)

    elif method == "industry":
        relevance_i = relevance.copy()
        relevance_i = sorted(relevance_i, reverse=True)
        relevance_i = 2 ** np.array(relevance_i)[:k] - 1
        denom_i = np.log2(np.arange(2, k + 2, dtype=int))
        score_i = relevance_i / denom_i

        relevance = 2 ** np.array(relevance)[:k] - 1
        denom = np.log2(np.arange(2, k + 2, dtype=int))
        score = np.sum((relevance / denom)) / np.sum(score_i)
    elif not isinstance(5, str):
        raise ValueError('')
    return score
