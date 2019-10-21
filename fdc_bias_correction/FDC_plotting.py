import pandas as pd
import argparse
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("observed_csv", type=str)
    parser.add_argument("fitted_csv", type=str)
    parser.add_argument("--estimated_csv", type=str)
    return parser.parse_args()


args = parse_arguments()
observed_df = pd.read_csv(args.observed_csv)
fitted_df = pd.read_csv(args.fitted_csv)

observed_df.drop(columns=["Unnamed: 0"], inplace=True)
fitted_df.drop(columns=["Unnamed: 0"], inplace=True)

assert len(observed_df) == len(fitted_df), "Inputs do not have the same number of rows"

probabilities = list(map(float, observed_df.columns[1:]))

for station_id in observed_df.index:
    observed_data = observed_df.loc[station_id, :].values
    fitted_data = fitted_df.loc[station_id, :].values

    assert fitted_data[0] == observed_data[0], "different_ordering?"
    plt.title("Reach ID {}".format(fitted_data[0]))
    plt.plot(probabilities, observed_data[1:])
    plt.plot(probabilities, fitted_data[1:])
    plt.legend(["Observed data", "Fitted data"])
    plt.show()