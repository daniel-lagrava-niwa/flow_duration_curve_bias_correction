import numpy as np
import pandas as pd
import xarray as xr
import argparse

import DistributionFitter
import DistributionSelector
import PreProcessing
import Visualization


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_nc", help="netCDF file containing the river flow")
    parser.add_argument("--sample_data", action='store_true')
    parser.add_argument("--create-plots", action='store_true')
    args = parser.parse_args()
    return args


args = parse_arguments()
DS = xr.open_dataset(args.input_nc)

stations = DS.coords["station"]
reach_ids = DS.station_rchid
f = open("original_values.csv", "w")
df = pd.DataFrame(columns=["reach_id", "distribution", "scale_mean", "scale_std", "param0","param1","param2"])

distribution = ["genextreme", "lognorm", "pearson3"]

current_station = 0
for station in stations:
    reach_id = int(reach_ids[station])
    print(reach_id)
    values = PreProcessing.select_complete_years(station, DS)
    if values is None:
        print("Nothing to be done, not enough values")
        continue

    if args.sample_data:
        values = PreProcessing.sample_data(values, size=1000)

    values_std, mean, std = PreProcessing.scale_data(values)

    f.write(",".join(list(map(str, values_std))) + "\n")

    fitter = DistributionFitter.DistributionFitter(values_std, distributions=[distribution])
    fitter.fit()
    fitted_params = fitter.fitted_parameters
    statistical_tests = DistributionSelector.compute_statistical_tests(values_std, fitted_params)
    best_dist_row = statistical_tests.loc[statistical_tests.aic == statistical_tests['aic'].min()]
    best_dist = best_dist_row['distribution'].values[0]
    if args.create_plots:
        Visualization.plot_fit_vs_data(values_std, best_dist, fitted_params[best_dist])

    df.at[current_station, ["reach_id","distribution", "scale_mean", "scale_std"]] = [reach_id, best_dist, mean, std]
    if len(fitted_params[best_dist]) == 3:
        df.at[current_station, ["param0", "param1", "param2"]] = [*fitted_params[best_dist]]
    else:
        df.at[current_station, ["param0", "param1", "param2"]] = [None, *fitted_params[best_dist]]

    current_station += 1
    print(df)

output_file_name = "%s.csv" % distribution
df.to_csv(output_file_name)
