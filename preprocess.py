# Author: SimpleSaad

import pandas as pd
import numpy as np
import os
path = os.path.dirname(__file__)
path1 = os.path.join(path, 'dataset/processed_cleveland.csv')
path2 = os.path.join(path, 'dataset/processed_hungarian.csv')
path3 = os.path.join(path, 'dataset/processed_switzerland.csv')
path4 = os.path.join(path, 'dataset/processed_va.csv')

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df3 = pd.read_csv(path3)
df4 = pd.read_csv(path4)

df1 = df1.replace('?', np.nan)
df2 = df2.replace('?', np.nan)
df3 = df3.replace('?', np.nan)
df4 = df4.replace('?', np.nan)

col = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df1.columns = col
df2.columns = col
df3.columns = col
df4.columns = col

print("Cleveland data. Size={}\nNumber of missing values".format(df1.shape))
print(df1.isna().sum())

print("\nHungary data:. Size={}\nNumber of missing values".format(df2.shape))
print(df2.isna().sum())

print("\nSwitzerland data. Size={}\nNumber of missing values".format(df3.shape))
print(df3.isna().sum())

print("\nV.A Long Beach data. Size={}\nNumber of missing values".format(df4.shape))
print(df4.isna().sum())

df = pd.concat([df1, df2, df3, df4])
df=df.fillna(df.median())

df=df.drop(['oldpeak', 'slope','ca', 'thal'], axis=1)
print("Concatanated dataset. Size={}\nNumber of missing values".format(df.shape))
print(df.isna().sum())

df.to_csv(os.path.join(path, 'recons_dataset/combined_dataset.csv'), index=False)
