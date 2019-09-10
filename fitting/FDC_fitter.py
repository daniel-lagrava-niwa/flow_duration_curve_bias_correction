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
    parser.add_argument("input_nc", help="netCDF file containing the river flow", optional=False)
    parser.add_argument("--create-plots")
    args = parser.parse_args()
    return args

