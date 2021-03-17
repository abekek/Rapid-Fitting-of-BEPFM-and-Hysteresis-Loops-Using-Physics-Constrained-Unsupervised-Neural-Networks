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


def loop_fitting_function(type, V, y):

    if(type == '9 parameters'):
        a0 = y[:, 0]
        a1 = y[:, 1]
        a2 = y[:, 2]
        a3 = y[:, 3]
        a4 = y[:, 4]
        b0 = y[:, 5]
        b1 = y[:, 6]
        b2 = y[:, 7]
        b3 = y[:, 8]
        d = 1000

        g1 = (b1 - b0) / 2 * (special.erf((V - a2) * d) + 1) + b0
        g2 = (b3 - b2) / 2 * (special.erf((V - a3) * d) + 1) + b2

        y1 = (g1 * special.erf((V - a2) / g1) + b0) / (b0 + b1)
        y2 = (g2 * special.erf((V - a3) / g2) + b2) / (b2 + b3)

        f1 = a0 + a1 * y1 + a4 * V
        f2 = a0 + a1 * y2 + a4 * V

        loop_eval = np.hstack((f1, f2))
        return loop_eval
    elif(type == '13 parameters'):
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

        # See supporting information for more information about the form of this function
        S1 = ((b1 + b2) / 2) + ((b2 - b1) / 2) * special.erf((V - b7) / b5)
        S2 = ((b4 + b3) / 2) + ((b3 - b4) / 2) * special.erf((V - b8) / b6)
        Branch1 = (a1 + a2) / 2 + ((a2 - a1) / 2) * \
            special.erf((V - Au) / S1) + a3 * V
        Branch2 = (a1 + a2) / 2 + ((a2 - a1) / 2) * \
            special.erf((V - Al) / S2) + a3 * V

        return np.concatenate((Branch1, np.flipud(Branch2)), axis=0).squeeze()
    else:
        print('No such parameters')
        return None


def loop_fitting_function_tf(type, V, y):
    if(type == '9 parameters'):
        a0 = y[:, :, 0]
        a1 = y[:, :, 1]
        a2 = y[:, :, 2]
        a3 = y[:, :, 3]
        a4 = y[:, :, 4]
        b0 = y[:, :, 5]
        b1 = y[:, :, 6]
        b2 = y[:, :, 7]
        b3 = y[:, :, 8]
        d = 1000

        g1 = tf.add(tf.multiply(tf.divide(tf.subtract(b1, b0), 2), tf.add(tf.multiply(tf.math.erf(tf.subtract(V, a2)), d), 1)), b0)
        g2 = tf.add(tf.multiply(tf.divide(tf.subtract(b3, b2), 2), tf.add(tf.multiply(tf.math.erf(tf.subtract(V, a3)), d), 1)), b2)

        y1 = tf.divide(tf.add(tf.multiply(g1, tf.math.erf(tf.divide(tf.subtract(V, a2), g1))), b0), tf.add(b0, b1))
        y2 = tf.divide(tf.add(tf.multiply(g2, tf.math.erf(tf.divide(tf.subtract(V, a3), g2))), b2), tf.add(b2, b3))

        f1 = tf.add(a0, tf.add(tf.multiply(a1, y1), tf.multiply(a4, V)))
        f2 = tf.add(a0, tf.add(tf.multiply(a1, y2), tf.multiply(a4, V)))

        return tf.transpose(tf.concat([f1, f2], axis=0))

    elif(type == '13 parameters'):

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
