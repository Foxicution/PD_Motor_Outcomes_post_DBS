#https://github.com/smazzanti/mrmr

from mrmr import mrmr_classif
import pandas as pd

df = pd.read_csv("Final_Data_index.csv", index_col=0)
df = df.dropna(axis=1)
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
selected_features = mrmr_classif(X=X, y=y, K=20)
selected_features.append('Efektas DBS (1-blogas, 2-geras, 3-labai geras)')
print(selected_features)

df = df[selected_features]

df.to_csv('MRMR2.csv')