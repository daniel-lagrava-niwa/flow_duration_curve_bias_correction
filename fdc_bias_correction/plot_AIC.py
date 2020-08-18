import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import pandas as pd
import argparse
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score


large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")


lognorm = pd.read_csv('lognorm.csv')
genextreme = pd.read_csv('genextreme.csv')
gumbel = pd.read_csv('gumbel_r.csv')
pareto  = pd.read_csv('genpareto.csv')

df = pd.concat([lognorm, genextreme, gumbel, pareto], ignore_index=True)
df.rename(columns={"aic": "AIC", "distribution": "Distribution"},  inplace=True)


plt.figure(figsize=(10, 13))
plt.yticks(fontsize=12, alpha=.7)
plt.title("AIC for several distributions", fontsize=22)

sns.boxplot(x="Distribution", y="AIC", data=df, whis=True)
sns.swarmplot(x="Distribution", y="AIC", data=df,
              size=2, color=".3", linewidth=0)
plt.savefig("AIC_2.png")
plt.close()
