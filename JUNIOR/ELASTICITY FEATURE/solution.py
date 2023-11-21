import pandas as pd
import numpy as np
from sklearn.metrics import r2_score as r2
from sklearn.linear_model import LinearRegression as lr


def elasticity_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
    df (pd.DataFrame):

    Returns:
    pd.DataFrame:

    """

    def fit_predict(x, q):
        """
        Args:
        P (pd.Series):
        q (pd.Series):

        Returns:
        pd.Series:
        """
        x = x.values.reshape(-1, 1)
        model = lr()
        model.fit(x, q)
        preds = model.predict(x)
        return preds

    result = df.groupby(by=["sku"]).apply(
        lambda x: r2(
            np.log(x["qty"] + 1), fit_predict(x["price"],
                                              np.log(x["qty"] + 1))))
    result = result.reset_index(drop=False)
    result.columns = ["sku", "elasticity"]

    return result
