import scipy.stats
import numpy as np
import sys
import os


def aic(data, distribution, params_predicted):
    LLH = distribution.logpdf(data, *params_predicted).sum()
    print(distribution, params_predicted, LLH)
    k = len(params_predicted)
    return 2 * k - 2 * LLH


def chi_squared(y, y_pred, k):
    pass


class DistributionFitter:
    """
    Class DistributionFitter
    """

    def __init__(self, values, distributions=None, best_criterion="AIC"):
        self.values = values
        self.distributions = ["genextreme", "lognorm", "gompertz"]
        if distributions is not None:
            self.distributions = distributions
        self.parameters = {}
        self.fit_failed = True
        assert best_criterion in ["AIC", "chi-squared", "KS"], "Invalid criterion: %s" % best_criterion
        self.best_criterion = best_criterion

    def fit(self):
        for distribution in self.distributions:
            dist = getattr(scipy.stats, distribution)
            params = dist.fit(self.values)
            self.parameters[distribution] = params
            self.fit_failed = False

    def get_best(self):
        if self.fit_failed:
            return None

        for dist, params in self.parameters.items():
            current_dist = getattr(scipy.stats, dist)
            if self.best_criterion == "AIC":
                AIC_value = aic(self.values, current_dist, params)
                print(AIC_value)


if __name__ == "__main__":
    s = 0.1
    values = scipy.stats.lognorm.rvs(s, size=1000)
    fitter = DistributionFitter(values)
    fitter.fit()
    fitter.get_best()
