"""
Created on Sun Jan 24 16:34:00 2021
@author: Alibek Kaliyev
"""

import numpy as np


class global_scaler:

    def fit(self, data):
        self.mean = np.mean(data.reshape(-1))
        self.std = np.std(data.reshape(-1))

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        return (data - self.mean)/self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
