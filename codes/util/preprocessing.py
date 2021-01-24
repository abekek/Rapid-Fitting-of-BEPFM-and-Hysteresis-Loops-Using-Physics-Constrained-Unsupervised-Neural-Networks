import numpy as np


class my_scaler:

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


def transform_params(params_real, params_pred):
    params_pred[:, 3] = params_pred[:, 3] + 1/2 * np.pi
    params_real[:, 3] = params_real[:, 3] + 1/2 * np.pi

    params_pred[:, 3] = np.where(
        params_pred[:, 3] < np.pi, params_pred[:, 3], params_pred[:, 3] - 2 * np.pi)
    params_real[:, 3] = np.where(
        params_real[:, 3] < np.pi, params_real[:, 3], params_real[:, 3] - 2 * np.pi)

    params_pred[:, 3] = np.where(
        params_pred[:, 2] > 0, params_pred[:, 3], params_pred[:, 3] + np.pi)
    params_real[:, 3] = np.where(
        params_real[:, 2] > 0, params_real[:, 3], params_real[:, 3] + np.pi)

    params_pred[:, 2] = np.where(
        params_pred[:, 2] > 0, params_pred[:, 2], params_pred[:, 2]*-1)
    params_real[:, 2] = np.where(
        params_real[:, 2] > 0, params_real[:, 2], params_real[:, 2]*-1)

    params_pred[:, 3] = np.where(
        params_pred[:, 3] < np.pi, params_pred[:, 3], params_pred[:, 3] - 2 * np.pi)
    params_real[:, 3] = np.where(
        params_real[:, 3] < np.pi, params_real[:, 3], params_real[:, 3] - 2 * np.pi)

    return params_real, params_pred
