import pandas as pd
import math


def limit_gmv(df: pd.DataFrame) -> pd.DataFrame:
    df_c = df.copy()
    gmv = []
    for row in df_c.itertuples():
        gmv_max = row[3] * row[4]
        if row[4] == 0 or row[2] == 0 or row[3] == 0:
            gmv.append(float(0))
        elif gmv_max <= row[2]:
            gmv.append(float(gmv_max))
        elif gmv_max > row[2] and (row[2] % row[3]) == 0:
            gmv.append(float(row[2]))
        else:
            gmv.append(float(math.floor((row[2] / row[3])) * row[3]))

    df_c['gmv'] = gmv
    return df_c
