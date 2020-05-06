# Warping tutorial

## B_example_warping

#### April 2020

#### Eva Chamorro

## Import packages
import os
import matplotlib
matplotlib.use('TkAgg')
#% matplotlib inline ## activate in jupyter notebook
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from warping_functions import *
from time_frequency_analysis_functions import *

### Load simulated signal
data = sio.loadmat(os.getcwd() + '/sig_pek_for_warp.mat')

s_t_dec = data['s_t_dec']
fs = data['fs']
r = data['r']
c1 = data['c1']

'''
 #s_t: propagated modes in a Pekeris waveguide with parameters
    #c1, c2, rho1, rho2: sound speed / density
    #D: depth
    #r: range
    #zs, zr: source/receiver depth
  #s_t_dec: same than s_t, except that time origin has been set for warping
  #fs: sampling frequency

  #NB: one can run optional_create_simulated_signal.m to generate another
  #simulated signal

'''

### Plot time series
# The first sample of s_t_dec corresponds to time r/c1

# Make the signal shorter, no need to keep samples with zeros
N_ok = 150;
s_ok = s_t_dec[0, 0:N_ok]
s_ok = s_ok[np.newaxis, :]

# Corresponding time and frequency axis
time = np.arange(1, N_ok + 1) / fs

# Now, let's have a look at it
plt.figure(figsize=(14, 10))
plt.plot(time[0, :], s_ok[0, :])
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude (A.U.)')
plt.title('Signal in the time domain')
plt.grid()
plt.show()

print('The code has loaded a signal propagated in a Pekeris waveguide')
print('and shows the time series')
input('Press ENTER to proceed with warping')

### Warping

# The warped signal will be s_w
s_w, fs_w = warp_temp_exa(s_ok, fs, r, c1)
M = len(s_w)
time_w = np.arange(0, M) / fs_w  ## time axis of the warped signal

plt.figure(figsize=(14, 10))
plt.plot(time_w[0, :], s_w[:, 0])
plt.xlabel('Warped time (sec)')
plt.ylabel('Amplitude (A.U.)')
plt.title('Warped signal')
plt.grid()
plt.show()

print('This is the time series of the warped signal')
print('It is a stretched version on the original signal')
input('Press ENTER to proceed with spectrograms')

### Spectrograms
### Original signal

# STFT computation
NFFT = 1024
N_window = 31  # you need a short window to see the modes

b = np.arange(1, N_ok + 1)
b = b[np.newaxis, :]
d = np.hamming(N_window)
d = d[:, np.newaxis]

tfr = tfrstft(s_ok, b, NFFT, d)
spectro = abs(tfr) ** 2

# Figure
freq = (np.arange(0, NFFT)) * fs / NFFT
plt.figure(figsize=(15.0, 10.0))
plt.subplot(121)
plt.pcolormesh(time[0, :], freq[0, :], spectro)
plt.ylim([0, fs / 2])
plt.xlim([0, 0.5])  ### Adjust this to see better
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Original signal')

N_window_w = 301;  ### You need a long window to see the warped modes
wind = np.hamming(N_window_w)
wind = wind / np.linalg.norm(wind)
wind = wind[:, np.newaxis]
t = np.arange(1, M + 1)
t = t[np.newaxis, :]

tfr_w = tfrstft(s_w, t, NFFT, wind)

spectro_w = abs(tfr_w) ** 2

### Frequency axis of the warped signal
freq_w = np.arange(0, NFFT) * fs / NFFT

### Figure
plt.subplot(122)
plt.pcolormesh(time_w[0, :], freq_w[0, :], spectro_w)
plt.ylim([0, 40])
plt.xlabel('Warped time (sec)')
plt.ylabel('Corresponding warped frequency (Hz)')
plt.title('Warped signal')
plt.show()

print('Here are the spectrograms')
print('Note that window length is highly different for')
print('the spectrogram of the original signal and the ')
print('spectrogram of the warped signal (look at codes and comments)')
input('Press ENTER to proceed with inverse warping')

## Let's unwarp the signal
s_r = iwarp_temp_exa(s_w, fs_w, r, c1, fs, N_ok)

plt.figure(figsize=(14, 10))
plt.plot(time[0, :], s_ok[0, :])
plt.plot(time[0, :], s_r[:, 0], 'or', fillstyle='none')
plt.grid()
plt.xlabel('Time (sec')
plt.title('blue: original ; red: after warping + unwarping')
plt.show()

