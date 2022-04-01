#Paulius Lapienis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import scipy.stats as stats

def calculate_pvalues_S(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = stats.spearmanr(df[r], df[c])[1]
    return pvalues, df.shape[0]

def calculate_pvalues_P(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = stats.pearsonr(df[r], df[c])[1]
    return pvalues, df.shape[0]


df = pd.read_csv("Final_Data_index.csv", index_col=0)
df = df.replace(r'^\s*$', np.nan, regex=True)

df.to_csv("Fixed_Final_Data_index.csv")
df = pd.read_csv("Fixed_Final_Data_index.csv", index_col=0)
df['Efektas DBS (1-blogas, 2-geras, 3-labai geras)'].replace({3: 2}, inplace=True)
print(df)

##############################################################################
## Calculating features with sig dif between populations ##

from scipy.stats import f_oneway, kruskal, mannwhitneyu

outcome = ['Efektas DBS (1-blogas, 2-geras, 3-labai geras)', 'Psicho_komplikacijos']

O_DBS = pd.DataFrame(columns=['feature', 'N_G1', 'N_G2', 'N', 'p-value'])

for i in df.keys():
    if i != 'Efektas DBS (1-blogas, 2-geras, 3-labai geras)':
        temp = df[['Efektas DBS (1-blogas, 2-geras, 3-labai geras)', i]].dropna()
        CategoryGroupLists=temp.groupby('Efektas DBS (1-blogas, 2-geras, 3-labai geras)')[i].apply(list)
        try:
            g1 = len(CategoryGroupLists[1])
            g2 = len(CategoryGroupLists[2])
            N = g1 + g2
            p = kruskal(*CategoryGroupLists)[1]
            if N >= 10 and p < 0.05:
                a_series = pd.Series([i, g1, g2, N, p], index = O_DBS.columns)
                O_DBS = O_DBS.append(a_series, ignore_index=True)
                print(i, g1, g2, N, p)
        except:
            print("An empty group is present", i)
        # print(*CategoryGroupLists)
O_DBS.to_csv("DBS_Outcome_Correlations.csv")

#Picking features with maximum number of patients
O_DBS = O_DBS[O_DBS.N == O_DBS.N.max()]

features = O_DBS['feature'].tolist()
features.append('Efektas DBS (1-blogas, 2-geras, 3-labai geras)')

f_df = df[features]

f_df.to_csv("Temp.csv")

###############################################################################

## Checking for redundancy ##
# Very unoptimized -_-

features = pd.read_csv("Temp.csv", index_col=0)
features = features.drop(['Efektas DBS (1-blogas, 2-geras, 3-labai geras)'], axis=1)
redundancy = features.corr(method="spearman").abs()
flattened = redundancy.unstack().drop_duplicates()
indexes = [index for index in flattened.index]
for index in indexes:
    if index[0] == index[1]:
        print("x")
        flattened = flattened.drop(index, axis = 0)
print(flattened.shape)
flattened = flattened[flattened > 0.5]
print(flattened.shape)
flattened.to_csv("Redundancy.csv")

###############################################################################

## Removing redundant features ##
O_DBS = pd.read_csv("DBS_Outcome_Correlations.csv", index_col=0)
flattened = pd.read_csv("Redundancy.csv", index_col=(0, 1))
f_df = pd.read_csv("Temp.csv", index_col=0)
for index in flattened.index:
    # print(index[0])
    # print(O_DBS.loc[O_DBS['feature'] == index[0]]['p-value'].iloc[0])
    if (O_DBS.loc[O_DBS['feature'] == index[0]]['p-value'].iloc[0] <
        O_DBS.loc[O_DBS['feature'] == index[1]]['p-value'].iloc[0]):
        if index[0] in f_df.columns:
            f_df = f_df.drop(index[0], axis=1)
            print(index[0])
    else:
        if index[1] in f_df.columns:
            f_df = f_df.drop(index[1], axis=1)
            print(index[1])

f_df.to_csv("MRMR.csv")

###############################################################################