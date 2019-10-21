import argparse
import numpy as np
import pandas as pd
import PreProcessing
import yaml
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    return parser.parse_args()


args = parse_arguments()
stream = open(args.config_file)
config = yaml.load(stream)
print(config)

original_time_series_file = config['inputs']['original_data']
original_estimations_file = config['inputs']['original_estimations']
expected_values_file = config['inputs']['expected_estimations']
predicted_values_file = config['inputs']['predicted_estimations']

original_time_series = pd.read_csv(original_time_series_file)
original_estimations = pd.read_csv(original_estimations_file)
expected_values = pd.read_csv(expected_values_file)
predicted_values = pd.read_csv(predicted_values_file)

assert len(expected_values) == len(predicted_values), "Mismatching number of values in expected vs predicted"

encoded_distributions = list(filter(lambda x: x.startswith("distribution_"), predicted_values.columns))
distributions = list(map(lambda x: x.replace("distribution_",""), encoded_distributions))
print(encoded_distributions)

for entry in np.arange(len(expected_values)):
    expected_row = expected_values.iloc[entry]
    predicted_row = predicted_values.iloc[entry]

    expected_params = expected_row.loc[["param0","param1","param2"]].to_list()
    predicted_params = predicted_row.loc[["param0","param1","param2"]].to_list()

    index_expected = int(expected_row["Unnamed: 0"])
    reach_id = int(original_estimations.iloc[index_expected]["NZReach"])
    print(index_expected, reach_id)

    expected_dist = expected_row.loc[encoded_distributions].idxmax()
    predicted_dist = predicted_row.loc[encoded_distributions].idxmax()
    print(expected_dist, expected_params)
    print(predicted_dist, predicted_params)

    expected_dist = expected_dist.split("_")[1]
    predicted_dist = predicted_dist.split("_")[1]

    if predicted_params[0] < 0. and predicted_dist == "lognorm":
        print("there is an error with that parameter")
        continue

    expected_p, expected_FDC = PreProcessing.calculate_FDC_from_distribution(expected_dist, expected_params)
    predicted_p, predicted_FDC = PreProcessing.calculate_FDC_from_distribution(predicted_dist, predicted_params)

    original_values = original_time_series.iloc[index_expected].to_numpy()
    original_p, original_data = PreProcessing.calculate_FDC_from_data(original_values)

    plt.plot(original_p[:], original_data[:], label="original")
    plt.plot(expected_p[:], expected_FDC[:], label="expected")
    plt.plot(predicted_p[:], predicted_FDC[:], label="predicted")

    plt.legend()
    plt.show()
