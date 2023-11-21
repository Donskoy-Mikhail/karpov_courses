from typing import List, Tuple
from scipy import stats


def ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Two-sample t-test for the means of two independent samples"""
    result = False
    res_test = stats.ttest_ind(control, experiment)
    p_value = res_test.pvalue

    if p_value < alpha:
        result = True

    return p_value, result
