import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import pandas as pd
import argparse
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("observed_csv", type=str)
    parser.add_argument("fitted_csv", type=str)
    parser.add_argument("estimated_csv", type=str)
    parser.add_argument("--distribution", type=str, default="unknown")
    parser.add_argument("--output_dir", type=str, default=".")
    return parser.parse_args()


def plot_fdc(observed_df, fitted_df, predicted_df, reach_index, upper_prob=0.99, lower_prob=0.01, output_dir="."):
    # create fdc 90%-10%
    i = reach_index
    rchid = observed_df.iloc[i,0]
    probabilities = np.array(list(map(float, observed_df.columns[1:])))
    idx_1 = np.abs(probabilities - upper_prob).argmin() + 1
    idx_0 = np.abs(probabilities - lower_prob).argmin() + 1
    filename = os.path.join(output_dir, "fdc_%i_%.2f-%.2f.png" % (rchid, upper_prob, lower_prob))

    obs_values = observed_df.iloc[i, idx_0:idx_1].values
    fitted_values = fitted_df.iloc[i, idx_0:idx_1].values
    pred_values = predicted_df.iloc[i, idx_0:idx_1].values

    rchid = int(observed_df.iloc[i, 0])
    plt.figure(figsize=(16, 10))
    plt.yticks(fontsize=12, alpha=.7)
    plt.title("Flow Duration Curve (Reach Id: %i)" % rchid, fontsize=22)
    plt.grid(axis='both', alpha=.3)
    plt.plot(probabilities[idx_0:idx_1], obs_values, color='tab:red')
    plt.plot(probabilities[idx_0:idx_1], fitted_values, color='tab:blue')
    plt.plot(probabilities[idx_0:idx_1], pred_values, color='tab:green')
    plt.legend(["Observed", "Fitted", "Predicted"])

    plt.gca().spines["top"].set_alpha(0.0)
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)
    plt.gca().spines["left"].set_alpha(0.3)
    plt.savefig(filename)
    plt.close()


def gather_statistics(observed_df, fitted_df, predicted_df, upper_prob=0.99, lower_prob=0.01):
    df = pd.DataFrame(columns=["reach_id", "r2", "RMSE", "class"])
    counter = 0
    filename = os.path.join(output_dir, "r2_%.2f-%.2f.png" % (upper_prob, lower_prob))
    for i in np.arange(len(observed_df)):
        rchid = observed_df.iloc[i, 0]
        probabilities = np.array(list(map(float, observed_df.columns[1:])))
        idx_1 = np.abs(probabilities - upper_prob).argmin() + 1
        idx_0 = np.abs(probabilities - lower_prob).argmin() + 1

        obs_values = observed_df.iloc[i, idx_0:idx_1].values
        fitted_values = fitted_df.iloc[i, idx_0:idx_1].values
        pred_values = predicted_df.iloc[i, idx_0:idx_1].values

        # change the negative values to 0.0001
        fitted_values[np.where(fitted_values < 0.0)] = 0.0001
        pred_values[np.where(pred_values < 0.0)] = 0.0001

        try:
            fitted_r2 = r2_score(np.log(obs_values), np.log(fitted_values))
            predicted_r2 = r2_score(np.log(obs_values), np.log(pred_values))
            fitted_rmse = mean_squared_error(np.log(obs_values), np.log(fitted_values))
            predicted_rmse = mean_squared_error(np.log(obs_values), np.log(pred_values))
        except ValueError as v:
            print(i)
            continue

        if fitted_r2 < 0 or predicted_r2 < 0:
            continue
        if fitted_rmse > .2 or predicted_rmse > .2:
            continue

        df.loc[counter, ["reach_id", "r2", "RMSE", "class"]] = [rchid, fitted_r2, fitted_rmse, "fitted"]
        df.loc[counter + 1, ["reach_id", "r2", "RMSE", "class"]] = [rchid, predicted_r2, predicted_rmse, "predicted"]
        counter += 2
    return df


def plot_r2(df, filename, output_dir="."):
    plt.figure(figsize=(16, 10))
    sns.boxplot(x='class', y='r2', data=df)
    sns.stripplot(x='class', y='r2', data=df, color='black')

    plt.title('r2 error for fitted and predicted vs. observed FDC', fontsize=28)
    plt.savefig(os.path.join(output_dir,filename))
    plt.close()


def plot_rmse(df, filename, output_dir="."):
    plt.figure(figsize=(16, 10))
    sns.boxplot(x='class', y='RMSE', data=df)
    sns.stripplot(x='class', y='RMSE', data=df, color='black')

    plt.title('RMSE for fitted and predicted wrt. observed FDC', fontsize=28)
    plt.savefig(os.path.join(output_dir,filename))
    plt.close()



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


observed_df.drop(columns=["Unnamed: 0"], inplace=True)
fitted_df.drop(columns=["Unnamed: 0"], inplace=True)
predicted_df.drop(columns=["Unnamed: 0"], inplace=True)


# for i in np.arange(len(observed_df)):
#     plot_fdc(observed_df, fitted_df, predicted_df, i, output_dir=output_dir)
#     plot_fdc(observed_df, fitted_df, predicted_df, i, upper_prob=0.9, lower_prob=0.1, output_dir=output_dir)
#     plot_fdc(observed_df, fitted_df, predicted_df, i, upper_prob=0.8, lower_prob=0.2, output_dir=output_dir)

stats_total = gather_statistics(observed_df, fitted_df, predicted_df)
stats_0p9_0p1 = gather_statistics(observed_df, fitted_df, predicted_df, upper_prob=0.9, lower_prob=0.1)
stats_0p8_0p2 = gather_statistics(observed_df, fitted_df, predicted_df, upper_prob=0.8, lower_prob=0.2)

plot_r2(stats_total, "r2_full.png", output_dir=output_dir)
plot_r2(stats_0p9_0p1, "r2_0p9_0p1.png", output_dir=output_dir)
plot_r2(stats_0p8_0p2, "r2_0p8_0p2.png", output_dir=output_dir)

plot_rmse(stats_total, "rmse_full.png", output_dir=output_dir)
plot_rmse(stats_0p9_0p1, "rmse_0p9_0p1.png", output_dir=output_dir)
plot_rmse(stats_0p8_0p2, "rmse_0p8_0p2.png", output_dir=output_dir)

