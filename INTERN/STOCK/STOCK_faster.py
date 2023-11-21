import pandas as pd
import numpy as np


def limit_gmv(df: pd.DataFrame) -> pd.DataFrame:
    df_c = df.copy()

    # Создаем массив для GMV и заполняем его нулями
    gmv = np.zeros(len(df_c), dtype=float)

    # Индексы, где stock, price или gmv равны 0
    zero_indices = (df_c['stock'] == 0) | (df_c['price'] == 0) | (df_c['gmv'] == 0)

    # Индексы, где GMV_max <= stock
    max_indices = (df_c['price'] * df_c['stock']) <= df_c['gmv']

    # Заполняем GMV для случаев, когда stock, price или gmv равны 0
    gmv[zero_indices] = 0

    # Заполняем GMV для случаев, когда GMV_max <= stock
    gmv[max_indices] = df_c[max_indices]['price'] * df_c[max_indices]['stock']

    # Заполняем GMV для остальных случаев
    remaining_indices = ~zero_indices & ~max_indices
    gmv[remaining_indices] = np.floor(df_c[remaining_indices]['gmv'] / df_c[remaining_indices]['price']) * \
                             df_c[remaining_indices]['price']

    df_c['gmv'] = gmv
    return df_c
