import pandas as pd


def fillna_with_mean(df: pd.DataFrame, target: str, group: str) -> pd.DataFrame:
    filled_df = df.copy()

    filled_df[target] = (filled_df.groupby(group)[target]
                         .transform(lambda x: x.fillna(x.mean() // 1)))

    return filled_df
