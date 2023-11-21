import numpy as np
from typing import List
from typing import Tuple
from scipy.stats import ttest_ind


def bootstrapped_q(x: List[float], n_bootstraps: int = 10_000, quantile: float = 0.95) -> List[float]:
    """Bootstrapped median distribution"""
    bootstrapped_quantile = []

    for _ in range(n_bootstraps):
        bootstrapped_sample = np.random.choice(x, size=len(x), replace=True)
        bootstrapped_quantile.append(np.quantile(bootstrapped_sample, quantile))

    return bootstrapped_quantile


def quantile_ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
    quantile: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[float, bool]:
    """
    Bootstrapped t-test for quantiles of two samples.
    """
    result = False
    control_quantile = bootstrapped_q(control, n_bootstraps, quantile)
    experiment_quantile = bootstrapped_q(experiment, n_bootstraps, quantile)
    res_test = ttest_ind(control_quantile, experiment_quantile)
    p_value = res_test.pvalue

    if p_value < alpha:
        result = True
    return p_value, result
