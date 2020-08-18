import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import pandas as pd
import argparse
import numpy as np
import os
import pandas as pd
import xarray as xr


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("observed_csv", type=str)
    parser.add_argument("fitted_csv", type=str)
    parser.add_argument("estimated_csv", type=str)
    parser.add_argument("--distribution", type=str, default="unknown")
    parser.add_argument("--output_dir", type=str, default=".")
    return parser.parse_args()



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

args = parse_arguments()
observed_df = pd.read_csv(args.observed_csv)
fitted_df = pd.read_csv(args.fitted_csv)
predicted_df = pd.read_csv(args.estimated_csv)
output_dir = args.output_dir
print(predicted_df.describe())

#plt.figure()


probabilities_float = np.array([i/10. for i in np.arange(1,10)])
print(probabilities_float)

index_list = []
for probability in probabilities_float:
    probabilities = np.array(list(map(float, observed_df.columns[2:])))
    closest_index = np.abs(probabilities - probability).argmin()
    index_list.append(closest_index)

observed_values = observed_df.iloc[:,index_list]
fitted_values = fitted_df.iloc[:,index_list]
estimated_values = predicted_df.iloc[:,index_list]

error_fitted = np.abs(observed_values - fitted_values)/observed_values
error_estimated = np.abs(observed_values - estimated_values)/observed_values
print(error_fitted)
df = pd.DataFrame(columns=["Probability", "Fitted Error", "Estimated Error"])
current_index = 0
for i in np.arange(len(index_list)):
    # print(probabilities_float[i], error_fitted[i], error_estimated[i])
    # df.loc[current_index, ["Probability", "Fitted Error", "Estimated Error"]] = [probabilities_float[i], error_fitted[i], error_estimated[i]]
    current_index += 1

print(df)