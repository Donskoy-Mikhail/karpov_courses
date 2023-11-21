import pandas as pd
import numpy as np

data = {
    'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Category_d': ['Afe', 'Bseg', 'Abdt', 'Btdhh', 'dthA', 'Bthd'],
    'Value': [10, 15, np.nan, 80, 30, np.nan]
}

df = pd.DataFrame(data)
grouped = df[['Category', 'Value']].groupby('Category')
print(grouped.mean())
df['Value'] = grouped['Value'].transform(lambda x: x.fillna(x.mean().round(0)))

print(df)