import scipy.stats
import pandas as pd
import numpy as np


def calculate_aic(data, distribution, params):
    LLH = distribution.logpdf(data, *params).sum()
    k = len(params)
    return 2 * k - 2 * LLH


def calculate_chisquare(data, distribution, params):

    chisq, p = scipy.stats.chisquare(data)
    return p


def calculate_ks(data, distribution_str, params):
    D, p = scipy.stats.kstest(data, distribution_str, args=params)
    print(D, p)
    return p


def compute_statistical_tests(data, fitting_dict):
    df = pd.DataFrame(columns=["distribution", "aic", "chisquared", "KS"])
    current_idx = 0
    for distribution_str in fitting_dict.keys():
        distribution = getattr(scipy.stats, distribution_str)
        # perform all tests
        aic = calculate_aic(data, distribution, fitting_dict[distribution_str])
        chisquared = calculate_chisquare(data, distribution, fitting_dict[distribution_str])
        KS = calculate_ks(data, distribution_str, params=fitting_dict[distribution_str])

        # put results on the data frame
        df.at[current_idx, ["distribution", "aic", "chisquared", "KS"]] = [distribution_str, aic, chisquared, KS]
        current_idx += 1
    return df


if __name__ == "__main__":
    import DistributionFitter

    s = 0.1
    values = np.random.gamma(0.1, size=1000)
    fitter = DistributionFitter.DistributionFitter(values)
    fitter.fit()
    fitted_parameters = fitter.fitted_parameters

    statistical_tests = compute_statistical_tests(values, fitted_parameters)
    print(statistical_tests)
    best_dist = statistical_tests.loc[statistical_tests.aic == statistical_tests['aic'].min()]
    print(best_dist['distribution'])