"""
Created on Sun Jan 24 16:34:00 2021
@author: Alibek Kaliyev
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from os.path import join as pjoin
import glob
import moviepy as mpy
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from ..util.postprocessing import convert_real_imag


def make_movie(movie_name, input_folder, output_folder, file_format,
               fps, output_format='mp4', reverse=False):
    """
    Function which makes movies from an image series
    Parameters
    ----------
    movie_name : string
        name of the movie
    input_folder  : string
        folder where the image series is located
    output_folder  : string
        folder where the movie will be saved
    file_format  : string
        sets the format of the files to import
    fps  : numpy, int
        frames per second
    output_format  : string, optional
        sets the format for the output file
        supported types .mp4 and gif
        animated gif create large files
    reverse : bool, optional
        sets if the movie will be one way of there and back
    """

    # searches the folder and finds the files
    file_list = glob.glob(input_folder + '/*.' + file_format)

    # Sorts the files by number makes 2 lists to go forward and back
    list.sort(file_list)
    file_list_rev = glob.glob(input_folder + '/*.' + file_format)
    list.sort(file_list_rev, reverse=True)

    # combines the file list if including the reverse
    if reverse:
        new_list = file_list + file_list_rev
    else:
        new_list = file_list

    if output_format == 'gif':
        # makes an animated gif from the images
        clip = ImageSequenceClip(new_list, fps=fps)
        clip.write_gif(output_folder + '/{}.gif'.format(movie_name), fps=fps)
    else:
        # makes and mp4 from the images
        clip = ImageSequenceClip(new_list, fps=fps)
        clip.write_videofile(
            output_folder + '/{}.mp4'.format(movie_name), fps=fps)


# plots 5 worst and best reconstructions
def plot_best_worst_SHO(real_data, pred_data, highest, wvec_freq):
    fig, axs = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle('5 worst and best reconstructions', fontsize=20)

    i = 0
    for x in highest:
        magnitude_graph_real, phase_graph_real = convert_real_imag(np.atleast_2d(real_data[x, :]))
        magnitude_graph_pred, phase_graph_pred = convert_real_imag(np.atleast_2d(pred_data[x, :]))

        axs[0, i].plot(wvec_freq, magnitude_graph_real[0, :], 'o', markersize=4,
                        label='amplitude initial')
        axs[0, i].plot(wvec_freq, phase_graph_real[0, :], 's', markersize=4,
                        label='phase initial')
        ax2 = axs[0, i].twinx()
        ax2.plot(wvec_freq, magnitude_graph_pred[0, :], '-.', label='amplitude pred')
        ax2.plot(wvec_freq, phase_graph_pred[0, :], '-.', label='phase pred')
        ax2.set_title("#" + str(x))
        ax2.set(xlabel='Frequency (Hz)', ylabel='Amplitude (Arb. U.)')
        i += 1

    for i in range(5):
        x = np.random.randint(0, real_data.shape[0])
        magnitude_graph_real, phase_graph_real = convert_real_imag(np.atleast_2d(real_data[x, :]))
        magnitude_graph_pred, phase_graph_pred = convert_real_imag(np.atleast_2d(pred_data[x, :]))

        axs[1, i].plot(wvec_freq, magnitude_graph_real[0, :], 'o', markersize=4,
                        label='amplitude initial')
        axs[1, i].plot(wvec_freq, phase_graph_real[0, :], 's', markersize=4,
                        label='phase initial')
        ax2 = axs[1, i].twinx()
        ax2.plot(wvec_freq, magnitude_graph_pred[0, :], '-.', label='amplitude pred')
        ax2.plot(wvec_freq, phase_graph_pred[0, :], '-.', label='phase pred')
        ax2.set_title("#" + str(x))
        ax2.set(xlabel='Frequency (Hz)', ylabel='Amplitude (Arb. U.)')
        i += 1

    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 2.7), loc='upper right', borderaxespad=0.)
    fig.subplots_adjust(top=0.87)


# plots 5 worst and best hysteresis loops
def plot_best_worst_loops(voltage, scaled_loops_DNN, scaled_loops_DNN_trust, scaled_loops_, highest, lowest, num_pix=3600):
    fig, axs = plt.subplots(3, 5, figsize=(15, 10))
    fig.suptitle('5 worst, best, and random loops', fontsize=20)

    i = 0
    for x in highest:
        axs[0, i].plot(voltage, scaled_loops_DNN[x],
                       'r--', label='small DNN model')
        axs[0, i].plot(voltage, scaled_loops_DNN_trust[x], 'b--',
                       label='small DNN model with trust region')
        axs[0, i].plot(voltage, scaled_loops_[x], 'o', markersize=4,
                       label='real_loops_scaled (conventional fits)')
        axs[0, i].set_title("#" + str(x))
        axs[0, i].set(xlabel='Voltage (V)', ylabel='Amplitude (Arb. U.)')
        i += 1

    i = 0
    for x in lowest:
        axs[1, i].plot(voltage, scaled_loops_DNN[x],
                       'r--', label='small DNN model')
        axs[1, i].plot(voltage, scaled_loops_DNN_trust[x], 'b--',
                       label='small DNN model with trust region')
        axs[1, i].plot(voltage, scaled_loops_[x], 'o', markersize=4,
                       label='real_loops_scaled (conventional fits)')
        axs[1, i].set_title("#" + str(x))
        axs[1, i].set(xlabel='Voltage (V)', ylabel='Amplitude (Arb. U.)')
        i += 1

    for i in range(5):
        j = np.random.randint(0, num_pix)
        axs[2, i].plot(voltage, scaled_loops_DNN[j],
                       'r--', label='small DNN model')
        axs[2, i].plot(voltage, scaled_loops_DNN_trust[j], 'b--',
                       label='small DNN model with trust region')
        axs[2, i].plot(voltage, scaled_loops_[j], 'o', markersize=4,
                       label='real_loops_scaled (conventional fits)')
        axs[2, i].set_title("#" + str(j))
        axs[2, i].set(xlabel='Voltage (V)', ylabel='Amplitude (Arb. U.)')

    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 4.0), loc='upper right', borderaxespad=0.)
    fig.subplots_adjust(top=0.87)
