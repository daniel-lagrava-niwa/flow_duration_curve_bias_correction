from typing import Dict, Any

import scipy.stats
import pandas as pd
import numpy as np
import sys
import os


class DistributionFitter:
    """
    Class DistributionFitter
    """
    def __init__(self, values, distributions=None):
        self.values = values
        self.distributions = ["genextreme", "lognorm", "pearson3"]
        if distributions is not None:
            self.distributions = distributions
        self.fit_failed = True
        self.parameters = {}

    def fit(self):
        for distribution in self.distributions:
            dist = getattr(scipy.stats, distribution)
            params = dist.fit(self.values)
            self.parameters[distribution] = params
            self.fit_failed = False

    @property
    def fitted_parameters(self):
        if self.fit_failed:
            return None
        return self.parameters


if __name__ == "__main__":
    s = 0.1
    values = scipy.stats.lognorm.rvs(s, size=1000)
    fitter = DistributionFitter(values)
    fitter.fit()
    print(fitter.fitted_parameters)

