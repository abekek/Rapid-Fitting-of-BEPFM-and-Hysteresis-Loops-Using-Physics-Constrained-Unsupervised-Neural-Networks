"""
Created on Sun Jan 24 16:34:00 2021
@author: Alibek Kaliyev
"""

import numpy as np


class global_standard_scaler:

    def __init__(self):
        self.mean = 0
        self.std = 1

    def fit(self, data):
        self.mean = np.mean(data.flatten())
        self.std = np.std(data.flatten())
        print("mean = ", self.mean, "STD = ", self.std)

    def transform(self, data):
        data -= self.mean
        data /= self.std
        return data

    def fit_transform(self, data):
        self.fit(data)
        self.transform(data)
        return data

    def inverse_transform(self, data):
        data *= self.std
        data += self.mean
        return data
