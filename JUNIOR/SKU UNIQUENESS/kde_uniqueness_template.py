from sklearn.neighbors import KernelDensity
import numpy as np


def kde_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on KDE.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    # Fit a kernel density estimator to the item embedding space
    kde = KernelDensity().fit(embeddings)

    uniqueness = []
    for item in embeddings:
        uniqueness.append(1 / np.exp(kde.score_samples([item])[0]))

    return np.array(uniqueness)
