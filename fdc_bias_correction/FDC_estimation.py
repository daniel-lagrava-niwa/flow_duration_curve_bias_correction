import pandas as pd
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import utils.federate_data
import scipy.stats
import utils.PreProcessing
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("federated_data", help="CSV file with characteristics and parameters")
    parser.add_argument("--regressor", type=str, default="linear", choices=["linear", "rf"])
    args = parser.parse_args()
    return args


args = parse_arguments()
federated_data_file = args.federated_data
assert args.regressor in ["linear", "rf"], "Invalid regressor {}".format(args.regressor)

full_data = pd.read_csv(federated_data_file)
predicted_parameters = full_data.drop(columns=["param0", "param1", "param2"])
X = full_data[utils.federate_data.DEFAULT_LOGGED_CHARACTERISTICS]

for i in range(0, 3):
    selected_param = "param%i" % i
    print("Fitting", selected_param)
    Y = full_data[[selected_param]]
    Y_predicted = np.zeros(Y.values.shape)
    cross_validation_error = 0.
    for index in np.arange(len(Y)):
        X_out = X.loc[index, :].values.reshape(1, -1)
        Y_out = Y.loc[index, :].values
        X_in = X.drop(index=index)
        Y_in = Y.drop(index=index)
        regression = None # will contain the regressor
        if args.regressor == "linear":
            regression = LinearRegression()
        elif args.regressor == "rf":
            regression = RandomForestRegressor(n_estimators=500, criterion="mse", verbose=False, bootstrap=True)
            Y_in = Y_in.values
            Y_in.shape = (len(Y_in),)
        else:
            print("Unkown regressor {}".format(args.regressor))
            exit(1)
        regression.fit(X_in, Y_in)
        Y_out_predicted = regression.predict(X_out)
        Y_predicted[index] = Y_out_predicted
        cross_validation_error += (Y_out_predicted - Y_out) ** 2

    cross_validation_error = cross_validation_error / len(Y)

    #regression = LinearRegression()
    #regression.fit(X, Y)
    #print(regression.coef_)
    #Y_predicted = regression.predict(X)
    plt.plot(Y, "bo")
    plt.plot(Y_predicted, "rx")
    plt.legend(["actual value", "predicted value"])
    plt.title("Fitting of %s vs. %s regression" % (selected_param, args.regressor))
    #plt.show()

    print("CrossValidationError =", cross_validation_error)
    print("MSE =", mean_squared_error(Y, Y_predicted))
    print("R2 =", r2_score(Y, Y_predicted))

    predicted_parameters.loc[:, selected_param] = Y_predicted

probabilities = utils.PreProcessing.get_probabilities()
comparison_columns = ["reach_id"] + list(probabilities)
predicted_values_df = pd.DataFrame(columns=comparison_columns)

for i in np.arange(len(predicted_parameters)):
    dist_parameters = predicted_parameters.loc[i, ["param0", "param1", "param2"]].values
    distribution = predicted_parameters.loc[i, ["distribution"]].values[0]
    dist = getattr(scipy.stats, distribution)
    distribution_values = dist.ppf(probabilities, *dist_parameters)
    distribution_values = np.sort(distribution_values)[::-1]
    reach_id = predicted_parameters.loc[i, ["NZReach"]].values[0]
    predicted_values_df.loc[i, "reach_id"] = reach_id
    print(reach_id, distribution_values, *dist_parameters)
    predicted_values_df.loc[i, probabilities] = distribution_values

predicted_parameters.drop(columns=["Unnamed: 0"], inplace=True)
predicted_parameters.to_csv(
    os.path.basename(federated_data_file).split(".")[0] + "_{}_predicted.csv".format(args.regressor))
predicted_values_df.to_csv("predicted_{}_{}.csv".format(distribution, format(args.regressor)))
