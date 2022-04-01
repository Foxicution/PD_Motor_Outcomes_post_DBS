import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap

df = pd.read_csv("MRMR_2.csv", index_col=0)
df1 = pd.read_csv("G1.csv", index_col=0, usecols =[i for i in range(23)])
df2 = pd.read_csv("G2.csv", index_col=0, usecols =[i for i in range(23)])

# fig, axes = plt.subplots(5,4, figsize=(15, 20))
fig, axes = plt.subplots(5,4, figsize=(15, 20))

for i,el in enumerate(list(df.columns.values)[:-1]):
    a = df.boxplot(el, by='DBS motor outcome (1-poor, 2-good/very good)', ax=axes.flatten()[i])
    a.grid('on', which='major', linewidth=1)
    title = a.set_title("\n".join(wrap(el, 30)))
    title.set_y(1.05)
  # remove empty subplot
plt.tight_layout() 
plt.suptitle('')
# fig.delaxes(axes[5,2])
# fig.delaxes(axes[5,3])
plt.show()
fig.savefig('boxplots.tif', dpi=300, bbox_inches='tight', format='tif')

# for i in df.columns:
#     myFig = plt.figure();
#     boxplot = df.boxplot(column=[i],
#                           by='Efektas DBS (1-blogas, 2-geras, 3-labai geras)')
#     plt.suptitle('')
#     myFig.savefig('%s.png'%(i), format='png')
