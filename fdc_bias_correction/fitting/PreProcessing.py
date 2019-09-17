import scipy.stats
import numpy as np
import xarray as xr


def scale_data(data):
    mean = data.mean()
    std = data.std()
    scaled_data = (data - mean)/std
    return scaled_data, mean, std


def original_data_from_scaled(scaled_data, mean, std):
    return scaled_data * std + mean


def calculate_FDC_from_data(data):
    copy_data = data.copy() * -1.
    copy_data.sort()
    copy_data = -1. * copy_data
    probabilities = np.linspace(0, 1, len(copy_data))
    return probabilities, copy_data


def calculate_FDC_from_distribution(distribution, params, size=1000):
    dist = getattr(scipy.stats, distribution)
    random_values = dist.rvs(*params, size=size).sort()[::-1]
    probabilities = np.linspace(0, 1, len(random_values))
    return probabilities, random_values


def sample_data(original_data, size=1000):
    if len(original_data) <= size:
        return original_data.copy()

    sampled_values = np.zeros(size)
    sorted_original_data = np.copy(original_data).copy()[::-1]
    probabilities_original_data = np.linspace(0, 1, len(original_data))
    probabilities_sampled_values = np.linspace(0, 1, size)

    for i in np.arange(size):
        idx = (np.abs(probabilities_original_data - probabilities_sampled_values[i])).argmin()
        sampled_values[i] = sorted_original_data[idx]

    return sampled_values


def calculate_chebyshev_nodes(n):
    k = np.arange(1, n)
    a = 0.
    b = 1.
    x_k = (a + b) / 2. + (b - a) / 2. * np.cos(2. * k - 1 / (2. * n) * np.pi)

    return x_k.sort()[::-1]


def sample_data_chebyshev(original_data, size=1000):
    # Calculate the chebyshev nodes
    x_k = calculate_chebyshev_nodes(size)

    sampled_values = np.zeros(size)
    sorted_original_data = np.sort(original_data)[::-1]
    probabilities_original_data = np.linspace(0, 1, len(original_data))

    # Same algorithm as before
    for i in np.arange(size):
        idx = (np.abs(probabilities_original_data - x_k[i])).argmin()
        sampled_values[i] = sorted_original_data[idx]

    return x_k, sampled_values



def select_complete_years(station, dataset, max_missing_days=30, min_valid_years=6, start_year=1972, end_year=2015):
    valid_years = 0
    data = []
    for year in np.arange(start_year, end_year + 1):
        year_slice = slice('%i-01-01' % year, '%i-12-31' % year)
        yearly_data = dataset.sel(time=year_slice).river_flow_rate[:, station].to_masked_array()
        if len(yearly_data) - yearly_data.count() > max_missing_days * 24:
            continue
        if np.count_nonzero(np.where(yearly_data >= 1e+35)) > 0:
            continue

        data += list(yearly_data.data[np.where(yearly_data.mask != True)])
        valid_years += 1

    if valid_years >= min_valid_years:
        return np.array(data)

    return None


if __name__ == "__main__":
    # Testing the cleaning
    data = scipy.stats.norm.rvs(size=1000)
    scaled_data, mean, std = scale_data(data)
    original_data = original_data_from_scaled(scaled_data, mean, std)

    p, v = calculate_FDC_from_data(scaled_data)
    import matplotlib.pyplot as plt
    plt.plot(p, v)
    plt.show()

    p, v = calculate_FDC_from_distribution("lognorm", [0.1,0.0,0.25])
    plt.plot(p, v)
    plt.show()