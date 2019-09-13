import argparse
import xarray as xr
import numpy as np
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("original_input_nc")
    parser.add_argument("original_estimations")
    parser.add_argument("expected_values")
    parser.add_argument("predicted_values")

    return parser.parse_args()


args = parse_arguments()
original_time_series = xr.open_dataset(args.original_input_nc)
original_estimations = pd.read_csv(args.original_estimations)
expected_values = pd.read_csv(args.expected_values)
predicted_values = pd.read_csv(args.predicted_values)

