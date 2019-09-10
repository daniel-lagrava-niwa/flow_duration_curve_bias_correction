import scipy.stats
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler


def scale_data(data):
    """
    Use sklearn.StandardScaler to convert data to 0 mean 1 std

    Parameters
    ----------
    data Input data

    Returns
    -------
    scaled data, scaler object (for inverse transform)

    """
    scaler = StandardScaler()
    reshaped_data = data.copy().reshape(-1,1)
    scaler.fit(reshaped_data)
    scaled_data = scaler.transform(reshaped_data)
    scaled_data = scaled_data.flatten()
    return scaled_data, scaler


def original_data_from_scaled(scaled_data, scaler):
    return scaler.inverse_transform(scaled_data)


if __name__ == "__main__":
    data = scipy.stats.norm.rvs(size=100)
    print(data)
    scaled_data, scaler = scale_data(data)
    print(scaled_data)
    original_data = original_data_from_scaled(scaled_data, scaler)
    print(original_data)