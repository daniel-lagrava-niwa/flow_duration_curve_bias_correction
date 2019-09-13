import xarray as xr
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("estimated_fdc")
    parser.add_argument("site_characteristics")
    return parser.parse_args()


args = parse_arguments()

estimated_parameters = pd.read_csv(args.estimated_fdc)
all_site_characteristics = pd.read_csv(args.site_characteristics)

merged_df = pd.merge(estimated_parameters, all_site_characteristics, left_on="reach_id", right_on="NZReach")
merged_df.to_csv("merged.csv")

selected_characteristics = merged_df[["usPET","usLake","usArea","usHard","usAvTCold", "USAvgSlope", "usSolarRadSum", "ORDER"]]
selected_estimated_params = merged_df[["distribution","param0","param1","param2"]]

encoded_selected_estimated_params = pd.get_dummies(selected_estimated_params)

train_chars, test_chars, train_params, test_params = train_test_split(selected_characteristics, encoded_selected_estimated_params , test_size=0.1)

rf_regressor = RandomForestRegressor(n_estimators=20)
rf_regressor.fit(train_chars, train_params)

predicted_estimated_params = rf_regressor.predict(test_chars)
print(type(predicted_estimated_params))
print(predicted_estimated_params)
print(test_params)
df_predicted = pd.DataFrame(data=predicted_estimated_params, columns=test_params.columns)
df_predicted.to_csv("predicted_test.csv")
test_params.to_csv("expected_test.csv")

