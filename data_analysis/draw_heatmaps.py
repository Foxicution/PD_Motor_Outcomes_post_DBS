#Paulius Lapienis

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(spearmanr(df[r], df[c])[1], 4)
    return pvalues


plt.figure(figsize=(16, 6))
dataframe = pd.read_csv("MRMR_2.csv", index_col=0, usecols =[i for i in range(21)])
p_values = calculate_pvalues(dataframe)
# p_values[p_values < 0.05] = '*'
# p_values[p_values >= 0.05] = ''
p_values = np.where(p_values < 0.05, np.where(p_values < 0.01, '**', '*'), '')
np.fill_diagonal(p_values, '')
strings = p_values
results = dataframe.corr(method='spearman').to_numpy()

labels = (np.asarray(["{1:.2f}{0}".format(string, value)
                      for string, value in zip(strings.flatten(),
                                               results.flatten())])
         ).reshape(20, 20)

print(labels)

# Increase the size of the heatmap.
plt.figure(figsize=(16, 6))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(dataframe.corr(method='spearman'), vmin=-1, vmax=1, annot=labels, fmt='')

heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right') 
plt.savefig('heatmap.tif', dpi=300, bbox_inches='tight', format='tif')


