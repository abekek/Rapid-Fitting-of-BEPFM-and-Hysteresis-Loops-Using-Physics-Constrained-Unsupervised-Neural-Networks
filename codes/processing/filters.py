import numpy as np


def range_filter(data, ranges):
    """
    includes only data within a range of values as selected by the user.\n

    Parameters
    ----------
    data : numpy array
        array of loops
    ranges : array
        range of values to include

    Returns
    -------
    data : numpy array
        array of loops
    """
    # checks if data is 3 dimensions
    if data.ndim == 3:

        # manually removes values which are too high or too low
        for i in range(data.shape[0]):

            for j in range(data.shape[1]):

                # finds low and high values
                low = data[i, j] < min(ranges)
                high = data[i, j] > max(ranges)
                outliers = np.where(low + high)

                # removes found values and sets = nan
                data[i, j, outliers] = np.nan
    else:

        raise ValueError('Input data does not have a valid dimension')

    return data


def clean_interpolate(data, fit_type='spline'):
    """
    Function which removes bad data points

    Parameters
    ----------
    data : numpy, float
        data to clean
    fit_type : string  (optional)
        sets the type of fitting to use

    Returns
    -------
    data : numpy, float
        cleaned data
    """

    # sets all non finite values to nan
    data[~np.isfinite(data)] = np.nan
    # function to interpolate missing points
    data = interpolate_missing_points(data, fit_type)
    # reshapes data to a consistent size
    data = data.reshape(-1, data.shape[2])
    return data


def interpolate_missing_points(data, fit_type='spline'):
    """
    Interpolates bad pixels in piezoelectric hysteresis loops.\n
    The interpolation of missing points allows for machine learning operations

    Parameters
    ----------
    data : numpy array
        array of loops
    fit_type : string (optional)
        selection of type of function for interpolation

    Returns
    -------
    data_cleaned : numpy array
        array of loops
    """

    # reshapes the data such that it can run with different data sizes
    if data.ndim == 2:
        data = data.reshape(np.sqrt(data.shape[0]).astype(int),
                            np.sqrt(data.shape[0]).astype(int), -1)
        data = np.expand_dims(data, axis=3)
    elif data.ndim == 3:
        data = np.expand_dims(data, axis=3)

    # creates a vector of the size of the data
    point_values = np.linspace(0, 1, data.shape[2])

    # Loops around the x index
    for i in range(data.shape[0]):

        # Loops around the y index
        for j in range(data.shape[1]):

            # Loops around the number of cycles
            for k in range(data.shape[3]):

                if any(~np.isfinite(data[i, j, :, k])):

                    # selects the index where values are nan
                    ind = np.where(np.isnan(data[i, j, :, k]))

                    # if the first value is 0 copies the second value
                    if 0 in np.asarray(ind):
                        data[i, j, 0, k] = data[i, j, 1, k]

                    # selects the values that are not nan
                    true_ind = np.where(~np.isnan(data[i, j, :, k]))

                    # for a spline fit
                    if fit_type == 'spline':
                        # does spline interpolation
                        spline = interpolate.InterpolatedUnivariateSpline(point_values[true_ind],
                                                                          data[i, j, true_ind, k].squeeze())
                        data[i, j, ind, k] = spline(point_values[ind])

                    # for a linear fit
                    elif fit_type == 'linear':

                        # does linear interpolation
                        data[i, j, :, k] = np.interp(point_values,
                                                     point_values[true_ind],
                                                     data[i, j, true_ind, k].squeeze())

    return data.squeeze()
