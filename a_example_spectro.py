# Warping tutorial
# A_example_spectro

# April 2020
# Eva Chamorro - Daniel Zitterbart - Julien Bonnel

#--------------------------------------------------------------------------------------
## 1. Import packages
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from subroutines.time_frequency_analysis_functions import *


'''
Select the environment where you want to run the code (Python Console or Terminal).
To activate select 1, to deactivate select 0 
Both environment cannot be activated at the same time
If you are using the Python Console you will have to close the figures to continue running the code 

We recommend to run the code in the Terminal, this way you can see all the results (figures) you don't have 
to close the figures

'''

PythonConsole=0
Terminal=1

if Terminal:
    matplotlib.use("TkAgg")

if PythonConsole == Terminal:
    raise ValueError ('Both environment cannot be activated/deactivated at the same time')


#--------------------------------------------------------------------------------------
## 2.  Load simulated signal
data = sio.loadmat(os.getcwd() + '/sig_pek_for_warp.mat')
s_t = data['s_t']
fs = data['fs']

'''
# s_t: propagated modes in a Pekeris waveguide with parameters
#      c1, c2, rho1, rho2: sound speed / density
#      D: depth
#      r: range
#      zs, zr: source/receiver depth
# s_t_dec: same than s_t, except that time origin has been set for warping
# fs: sampling frequency

### NB: one can run optional_create_simulated_signal.m to generate another
### simulated signal
### IF YOU CHANGE RANGE, you must change the following variable which
### defines the bounds on the x-axis for all the plots
'''

xlim_plots = [6.5, 7.5]

#--------------------------------------------------------------------------------------
## 3. Compute time-frequency representation

# Signal length
N = np.size(s_t)

# Short Time Fourier Transform parameters
vec_t = np.arange(1, N + 1) # time samples where the STFT will be computed (i.e. positions of the sliding window)

NFFT = 2048  # FFT size
N_window = 31  # sliding window size (need an odd number)

# STFT computation
d = np.hamming(N_window)
tfr = tfrstft(s_t, vec_t, NFFT, d)

# Spectrogram ~ modulus STFT
spectro = abs(tfr) ** 2

# Time and frequency axis
time = np.arange(0, N) / fs
freq = np.arange(0, NFFT) * fs / NFFT

#--------------------------------------------------------------------------------------
## 4.  Plots

print('The code has loaded a signal propagated in a Pekeris waveguide')
print('and shows the time series')


plt.figure()
plt.plot(time[0, :], s_t[0, :])
plt.xlim(xlim_plots)
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude (A.U.)')
plt.title('Signal in the time domain')
plt.grid()

if PythonConsole:
    print('Close the figure to continue and see the spectrogram')
    plt.show(block=True)
    # plt.ion()
    # input('Press ENTER to continue')

if Terminal:
    plt.get_current_fig_manager().window.wm_geometry("600x400+0+0")
    plt.show(block=False)
    input('Press ENTER to continue and see the spectrogram')

print('\n' * 20)
print('This is the spectrogram, computed with a sliding window ')
print('of length Nw=31 samples (or 0.15 s , since the sampling frequency is 200 Hz')
print('As a reminder, the spectrogram is computed by dividing the original signal into')
print('segments of length Nw. Another important parameter for spectrogram computation is')
print('overlap. In this tutorial, the overlap between segment is Nw-1 samples, which is ')
print('the maximum overlap that one can obtain : the percentage of overlap is 100*(Nw-1)/Nw')
print('Such a high overlap will be handy for modal filtering')


plt.figure()
plt.pcolormesh(time[0, :], freq[0, :], spectro)
plt.xlim(xlim_plots)
plt.ylim([0, fs / 2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram')

if PythonConsole:
    print('Close the figure to continue')
    plt.show(block=True)
    # plt.ion()
    # input('Press ENTER to continue')

if Terminal:
    plt.get_current_fig_manager().window.wm_geometry("600x400+800+0")
    plt.show(block=False)
    input('Press ENTER to continue')


print('\n' * 20)
print('Now you can modify the window length')
print('Remember to give it in number of samples')
print('Number between 11 and 71 are reasonable tries')
print('(for coding reason, please use odd numbers only)')

while ((N_window != 999)):
    print('Input window length and press enter for spectrogram computation : ')
    N_window = int(input('(window length 0 to exit the code) '))

    if ((N_window == 0) or (N_window == 999)):
        print('  ')
        print('END')
        break

    if ((N_window != 999)):
        # STFT computation
        d = np.hamming(N_window)
        tfr = tfrstft(s_t, vec_t, NFFT, d)
        # Spectrogram ~ modulus STFT
        spectro = abs(tfr) ** 2
        plt.figure()
        plt.pcolormesh(time[0, :], freq[0, :], spectro)
        plt.ylim([0, fs / 2])
        plt.xlim(xlim_plots)
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram N_window: ' + str(N_window))
        if PythonConsole:
            print('Close the figure to continue')
            plt.show(block=True)
            # plt.ion()
            # input('Press ENTER to continue')

        if Terminal:
            plt.get_current_fig_manager().window.wm_geometry("600x400+800+800")
            plt.show(block=False)
            input('Press ENTER to continue')
