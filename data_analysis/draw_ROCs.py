#Paulius Lapienis

import numpy as np
import matplotlib.pyplot as plt
from statistics import stdev

base_fpr = np.linspace(0, 1, 101)
accs = np.load('ACC.npy')
sens = np.load('SENS.npy')
spes = np.load('SPEC.npy')
aucs = np.load('AUC.npy')
tprs = np.load('TPRS.npy')

def PRES(k, res_acc, res_sens, res_spec, res_AUC, tprs, axis, name):
    print(name)
    print("%1d %4.2f  ±%4.2f    %4.2f ±%4.2f   %4.2f ±%4.2f   %4.2f ±%4.2f" % (k, 100*sum(res_acc)/len(res_acc), 100*stdev(res_acc), 100*sum(res_sens)/len(res_sens), 100*stdev(res_sens),
          100*sum(res_spec)/len(res_spec), 100*stdev(res_spec), 100*sum(res_AUC)/len(res_AUC), 100*stdev(res_AUC)))
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    axis.plot(base_fpr, mean_tprs, 'b')
    axis.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    axis.plot([0, 1], [0, 1],'r--')
    axis.set_xlim([-0.01, 1.01])
    axis.set_ylim([-0.01, 1.01])
    axis.set_ylabel('True Positive Rate')
    axis.set_xlabel('False Positive Rate')
    axis.text(x = 0.5, y = 0.15, s="AUC = %4.2f ± %4.2f" % (sum(res_AUC)/len(res_AUC), stdev(res_AUC)))
    axis.set_title(name, fontsize=15)

fig, axes = plt.subplots(2,4, figsize=(20, 10))
axes = axes.ravel()

names = {0: 'a) Regularized logistic regression', 1:'b) Decision tree classifier',
         2: 'c) Linear discriminant analysis', 3: 'd) Naive Bayes classifier',
         4: 'e) Support vector machine', 5: 'Random Choice',
         6: 'f) Deep feed-forward neural network',
         7: 'g) One class support vector machine',8: 'h) Autoencoder'}

for i, el in enumerate(list([0, 1, 2, 3, 4, 6, 7, 8])):
    PRES(i, accs[el], sens[el], spes[el], aucs[el], tprs[el], axes[i], names[el])

fig.savefig('ROCs.tif', dpi=300, bbox_inches='tight', format='tif')