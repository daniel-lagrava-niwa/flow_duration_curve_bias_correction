import scipy
import numpy as np
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt


def plot_fit_vs_data(data, distribution, params, bins=100, save=False, file_name="plot.png"):
    fig = plt.figure(figsize=(12, 9))
    dist = getattr(scipy.stats, distribution)
    x = np.linspace(dist.ppf(0.01, *params), dist.ppf(0.99, *params))
    plt.plot(x, dist.pdf(x, *params), label="Fitted data")
    plt.hist(data, bins=bins, label="Input data", normed=True, color="blue", alpha=0.5)
    plt.title("Fitting input data to %s" % distribution)
    plt.legend()

    if save:
        plt.savefig(file_name)
        plt.close()
        return

    plt.show()
    plt.close()


def plot_qq(data, distribution, params, save=False, file_name="qq_plot.png"):
    dist = getattr(scipy.stats, distribution)

    data_quantiles = data.copy()
    data_quantiles.sort()
    theoretical_quantiles = dist.rvs(*params, size=len(data_quantiles))
    theoretical_quantiles.sort()

    fig = plt.figure(figsize=(8, 5))
    plt.plot(theoretical_quantiles, data_quantiles, "o")
    min_value = np.floor(min(min(theoretical_quantiles), min(data_quantiles)))
    max_value = np.ceil(max(max(theoretical_quantiles), max(data_quantiles)))
    plt.plot([min_value, max_value], [min_value, max_value], 'r--')
    plt.xlabel("%s theoretical quantiles" % distribution)
    plt.title("qq plot for %s" % distribution)

    if save:
        plt.savefig(file_name)
        plt.close()
        return

    plt.show()
    plt.ylabel("Data quantiles")
    plt.close()


def plot_fdc(data, distribution, params, save=False, file_name="FDC.png"):
    data_copy = np.sort(data)[::-1]
    point_number = len(data_copy)
    dist = getattr(scipy.stats, distribution)

    probabilities = np.linspace(0.99,0.01,num=point_number)
    distribution_values = dist.ppf(probabilities, *params)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(probabilities, data_copy)
    plt.plot(probabilities, distribution_values)

    plt.title("FDC from %s vs. Observed data" % distribution)

    if save:
        plt.savefig(file_name)
        plt.close()
        return

    plt.show()
    plt.close()


if __name__ == "__main__":
    import DistributionFitter

    s = 0.1
    values = scipy.stats.lognorm.rvs(s, size=10000)
    fitter = DistributionFitter.DistributionFitter(values)
    fitter.fit()
    fitted_parameters = fitter.fitted_parameters

    for distribution in fitted_parameters.keys():
        plot_fit_vs_data(values, distribution, fitted_parameters[distribution])
        plot_qq(values, distribution, fitted_parameters[distribution])
