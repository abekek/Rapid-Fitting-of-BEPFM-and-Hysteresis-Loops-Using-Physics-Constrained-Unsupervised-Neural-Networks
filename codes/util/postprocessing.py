"""
Created on Sun Jan 24 16:34:00 2021
@author: Alibek Kaliyev
"""

import numpy as np


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

def convert_real_imag(data, type_data='stacked'):
  magnitude = np.abs(data)
  phase = np.angle(data)
  return magnitude, phase