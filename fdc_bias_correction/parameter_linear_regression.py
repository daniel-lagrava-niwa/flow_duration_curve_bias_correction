import pandas as pd
import argparse
import constants
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("federated_data", help="CSV file with characteristics and parameters")
    args = parser.parse_args()
    return args


args = parse_arguments()
federated_data_file = args.federated_data

full_data = pd.read_csv(federated_data_file)
predicted_parameters = full_data.drop(columns=["param0", "param1", "param2"])
X = full_data[constants.DEFAULT_LOGGED_CHARACTERISTICS]

for i in range(0,3):
    selected_param = "param%i" % i
    print("Fitting", selected_param)
    Y = full_data[[selected_param]]
    regression = LinearRegression()
    regression.fit(X,Y)
    print(regression.coef_)
    Y_predicted = regression.predict(X)
    print(mean_squared_error(Y,Y_predicted))
    print(r2_score(Y, Y_predicted))

    plt.plot(Y, "bo")
    plt.plot(Y_predicted, "rx")
    plt.legend(["actual value", "predicted value"])
    plt.title("Fitting of %s vs. linear regression" % selected_param)
    plt.show()

    predicted_parameters.loc[:, selected_param] = Y_predicted

predicted_parameters.drop(columns=["Unnamed: 0"], inplace=True)
predicted_parameters.to_csv(os.path.basename(federated_data_file).split(".")[0] + "_predicted.csv")

