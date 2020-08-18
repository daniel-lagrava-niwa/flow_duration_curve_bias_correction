import scipy.stats
import numpy as np


def scale_data(data):
    mean = data.mean()
    std = data.std()
    scaled_data = (data - mean)/std
    return scaled_data, mean, std


def original_data_from_scaled(scaled_data, mean, std):
    return scaled_data * std + mean


def get_probabilities(n=1001):
    return np.linspace(0.01,0.99,n)


def sample_data(original_data, size=1001):
    if len(original_data) <= size:
        print("Problem: not enough data points")
        return original_data.copy()

    sampled_values = np.zeros(size)
    sorted_original_data = np.sort(original_data)[::-1]
    probabilities_original_data = np.linspace(0.01, 1, len(original_data))
    probabilities_sampled_values = np.linspace(0.01, 1, size)

    for i in np.arange(size):
        idx = (np.abs(probabilities_original_data - probabilities_sampled_values[i])).argmin()
        sampled_values[i] = sorted_original_data[idx]

    return sampled_values


def select_complete_years(station, dataset, max_missing_days=30, min_valid_years=6, start_year=1972, end_year=2015):
    valid_years = 0
    data = []
    for year in np.arange(start_year, end_year + 1):
        year_slice = slice('%i-01-01' % year, '%i-12-31' % year)
        yearly_data = dataset.sel(time=year_slice).river_flow_rate[:, station].to_masked_array()
        if len(yearly_data) - yearly_data.count() > max_missing_days * 24:
            continue
        if np.count_nonzero(np.where(yearly_data >= 1e+35)) > 0:
            print("Big values")
            continue

        data += list(yearly_data.data[np.where(yearly_data.mask != True)])
        valid_years += 1
    print("Valid years: %i" % valid_years)
    if valid_years >= min_valid_years:
        return np.array(data)

    return None

