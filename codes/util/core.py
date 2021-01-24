"""
Created on Sun Jan 24 16:34:00 2021
@author: Alibek Kaliyev
"""

import torch
import numpy as np
import tensorflow as tf
from scipy import special


def SHO_fit_func_torch(parms,
                       wvec_freq,
                       device='cpu'):
    Amp = parms[:, 0].type(torch.complex128)
    w_0 = parms[:, 1].type(torch.complex128)
    Q = parms[:, 2].type(torch.complex128)
    phi = parms[:, 3].type(torch.complex128)
    wvec_freq = torch.tensor(wvec_freq)

    Amp = torch.unsqueeze(Amp, 1)
    w_0 = torch.unsqueeze(w_0, 1)
    phi = torch.unsqueeze(phi, 1)
    Q = torch.unsqueeze(Q, 1)

    wvec_freq = wvec_freq.to(device)

    numer = Amp * torch.exp((1.j) * phi) * torch.square(w_0)
    den_1 = torch.square(wvec_freq)
    den_2 = (1.j) * wvec_freq.to(device) * w_0 / Q
    den_3 = torch.square(w_0)

    den = den_1 - den_2 - den_3

    func = numer / den

    return func


def loop_fitting_function(V, a1, a2, a3, b1, b2, b3, b4, b5, b6, b7, b8, Au, Al):

    # See supporting information for more information about the form of this function
    S1 = ((b1 + b2) / 2) + ((b2 - b1) / 2) * special.erf((V - b7) / b5)
    S2 = ((b4 + b3) / 2) + ((b3 - b4) / 2) * special.erf((V - b8) / b6)
    Branch1 = (a1 + a2) / 2 + ((a2 - a1) / 2) * \
        special.erf((V - Au) / S1) + a3 * V
    Branch2 = (a1 + a2) / 2 + ((a2 - a1) / 2) * \
        special.erf((V - Al) / S2) + a3 * V

    return np.concatenate((Branch1, np.flipud(Branch2)), axis=0).squeeze()


def loop_fitting_function_tf(V, y):

    V = data['Voltagedata_mixed'][0:len(data['Voltagedata_mixed']) // 2]

    a1 = y[:, 0]
    a2 = y[:, 1]
    a3 = y[:, 2]
    b1 = y[:, 3]
    b2 = y[:, 4]
    b3 = y[:, 5]
    b4 = y[:, 6]
    b5 = y[:, 7]
    b6 = y[:, 8]
    b7 = y[:, 9]
    b8 = y[:, 10]
    Au = y[:, 11]
    Al = y[:, 12]

    epsilon = 5e-14

    S1 = tf.add(tf.divide(tf.add(b1, b2) + epsilon, 2.0), tf.multiply(tf.divide(tf.subtract(b2,
                                                                                            b1) + epsilon, 2.0), tf.math.erf(tf.divide(tf.subtract(V, b7) + epsilon, b5))))
    S2 = tf.add(tf.divide(tf.add(b4, b3) + epsilon, 2.0), tf.multiply(tf.divide(tf.subtract(b3,
                                                                                            b4) + epsilon, 2.0), tf.math.erf(tf.divide(tf.subtract(np.flipud(V), b8) + epsilon, b6))))
    Branch1 = tf.add(tf.add(tf.divide(tf.add(a1, a2) + epsilon, 2.0), tf.multiply(tf.divide(tf.subtract(
        a2, a1) + epsilon, 2.0), tf.math.erf(tf.divide(tf.subtract(V, Au) + epsilon, S1)))), tf.multiply(a3, V))
    Branch2 = tf.add(tf.add(tf.divide(tf.add(a1, a2) + epsilon, 2.0), tf.multiply(tf.divide(tf.subtract(a2, a1) + epsilon,
                                                                                            2.0), tf.math.erf(tf.divide(tf.subtract(np.flipud(V), Al) + epsilon, S2)))), tf.multiply(a3, np.flipud(V)))

    return tf.transpose(tf.concat([Branch1, Branch2], axis=0))


def computeDotProducts(u, v):
    return tf.reduce_sum(tf.stack([tf.reduce_sum(ui * vi) for ui, vi in zip(u, v)], 0))


def normOfVar(x):
    return tf.sqrt(computeDotProducts(x, x))
