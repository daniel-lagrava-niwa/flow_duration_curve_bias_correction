import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("merged_characteristics")
    return parser.parse_args()


args = parse_arguments()

federated_data = pd.read_csv(args.merged_characteristics)
print(federated_data)
fit_parameters = ["param%i" % i for i in [0,1,2]]

df_predicted = pd.DataFrame()

for i in np.arange(3):
    fit_candidate = "param%i" % i
    selected_estimated_params = federated_data[["param%i" % i]]
    to_drop = ["NZReach","distribution", "Unnamed: 0"] + fit_parameters
    selected_characteristics = federated_data.drop(columns=to_drop)
    print(selected_characteristics.columns)
    train_chars, test_chars, train_params, test_params = train_test_split(selected_characteristics,
                                                                          selected_estimated_params,
                                                                          test_size=0.1, random_state=0)

    print(train_params)
    print(test_params)
    rf_regressor = RandomForestRegressor(n_estimators=500, criterion="mse", verbose=True, bootstrap=True)
    rf_regressor.fit(train_chars, train_params)
    print("-------------------------------------------------------")
    print(selected_characteristics.columns)
    print(rf_regressor.feature_importances_)
    print(rf_regressor.score(train_chars, train_params))
    predicted_estimated_params = rf_regressor.predict(test_chars)

    # pd.crosstab(test_params, predicted_estimated_params, rownames=["Actual Result"], colnames=["Predicted Result"])

    # df_predicted = pd.DataFrame(data=predicted_estimated_params, columns=test_params.columns)
    df_predicted.loc[:, fit_candidate] = test_params[fit_candidate]
    df_predicted.loc[:, fit_candidate + "_predicted"] = np.array(predicted_estimated_params)

df_predicted.to_csv("predicted_test_param_lognorm.csv")

