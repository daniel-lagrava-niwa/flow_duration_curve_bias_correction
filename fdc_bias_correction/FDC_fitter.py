import numpy as np
import pandas as pd
import xarray as xr
import argparse
import logging
import glob
import os
import math

import fitting.DistributionFitter as DistributionFitter
import fitting.DistributionSelector as DistributionSelector
import fitting.PreProcessing as PreProcessing
import fitting.Visualization as Visualization


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", help="directory containing netCDF files containing the river flow")
    parser.add_argument("-considered_sites", help="file containing the sites to consider", default=None)
    parser.add_argument("--sample_data", action='store_true')
    parser.add_argument("--create_plots", action='store_true')
    args = parser.parse_args()
    return args


args = parse_arguments()

f = open("original_values.csv", "w")
df = pd.DataFrame(columns=["reach_id", "distribution", "param0","param1","param2"])

distribution = ["genextreme"]
current_station = 0

logging.basicConfig(filename="_".join(distribution) + ".log", level=logging.INFO )

sites_to_consider = pd.DataFrame()
if args.considered_sites is not None:
    print("Reading considered sites")
    sites_to_consider = pd.read_csv(args.considered_sites)

nc_files = glob.glob(os.path.join(args.input_directory, "*.nc"))

for nc_file in nc_files:
    print("FILE:", nc_file)
    DS = xr.open_dataset(nc_file)
    stations = DS.coords["station"]
    reach_ids = DS.station_rchid

    for station in stations:
        if math.isnan(reach_ids[station]):
            print("!!!!!!!", nc_file, station)
            continue

        reach_id = int(reach_ids[station])
        print(reach_id)

        if args.considered_sites is not None:
            if len(sites_to_consider.loc[sites_to_consider["NZReach"] == reach_id]) == 0:
                # print("%i is not on the list to be considered" % reach_id)
                logging.info("%i: not on selected sites" % reach_id)
                continue

        values = PreProcessing.select_complete_years(station, DS, min_valid_years=4, max_missing_days=60)
        if values is None:
            # print("%i Nothing to be done, not enough values" % reach_id)
            logging.info("%i: not enough data to continue" % reach_id)
            continue

        # Normalize by area
        upstream_area = sites_to_consider.loc[sites_to_consider.NZReach == reach_id]["usArea"].values[0]
        # log10_upstream_area = np.log10(upstream_area)
        # values_std = values / log10_upstream_area
        values_std = values / upstream_area

        if args.sample_data:
            values_std = PreProcessing.sample_data(values, size=101)

        # Writing the original values
        f.write(",".join(list(map(str, values_std))) + "\n")

        fitter = DistributionFitter.DistributionFitter(values_std, distributions=distribution)
        fitter.fit()
        fitted_params = fitter.fitted_parameters
        statistical_tests = DistributionSelector.compute_statistical_tests(values_std, fitted_params)

        # Select best distribution
        best_dist_row = statistical_tests.loc[statistical_tests.aic == statistical_tests['aic'].min()]
        best_dist = best_dist_row['distribution'].values[0]

        # Plotting (if required)
        if args.create_plots:
            file_name = "fit_%s_%i.png" % ("_".join(distribution), reach_id)
            Visualization.plot_fit_vs_data(values_std, best_dist, fitted_params[best_dist],
                                           save=True, file_name=file_name)

        df.at[current_station, ["reach_id","distribution"]] = [reach_id, best_dist]
        if len(fitted_params[best_dist]) == 3:
            df.at[current_station, ["param0", "param1", "param2"]] = [*fitted_params[best_dist]]
        else:
            df.at[current_station, ["param0", "param1", "param2"]] = [None, *fitted_params[best_dist]]

        logging.info("%i: succesfully processed" % reach_id)
        current_station += 1
        # print(df)

output_file_name = "_".join(distribution) + ".csv"
df.to_csv(output_file_name)
