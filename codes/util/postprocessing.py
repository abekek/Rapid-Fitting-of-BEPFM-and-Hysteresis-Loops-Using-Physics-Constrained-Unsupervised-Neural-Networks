import numpy as np


class Transformer:

    def __init__(self, params_real, params_pred):
        self.params_real = params_real
        self.params_pred = params_pred

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
