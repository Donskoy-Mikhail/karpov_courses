import pandas as pd
import numpy as np


def limit_gmv(df: pd.DataFrame) -> pd.DataFrame:
    df_c = df.copy()
    df_v = df.values

    # Создаем массив для GMV и заполняем его нулями
    gmv = np.zeros(df.shape[0], dtype=float)

    # Индексы, где stock, price или gmv равны 0
    zero_indices = (df_v[:, 3] == 0) | (df_v[:, 2] == 0) | (df_v[:, 1] == 0)

    # Индексы, где GMV_max <= stock
    max_indices = (df_v[:, 2] * df_v[:, 3]) <= df_v[:, 1]

    # Заполняем GMV для случаев, когда stock, price или gmv равны 0
    gmv[zero_indices] = 0

    # Заполняем GMV для случаев, когда GMV_max <= stock
    gmv[max_indices] = df_v[max_indices, 2] * df_v[max_indices, 3]

    # Заполняем GMV для остальных случаев
    remaining_indices = ~zero_indices & ~max_indices
    gmv[remaining_indices] = (np.floor(df_v[remaining_indices, 1] /
                                       df_v[remaining_indices, 2]) *
                              df_v[remaining_indices, 2])

    df_c['gmv'] = gmv
    return df_c
