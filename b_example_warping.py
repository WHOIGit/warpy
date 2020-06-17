# Warping tutorial
# B_example_warping

# April 2020
# Eva Chamorro - Daniel Zitterbart - Julien Bonnel

#--------------------------------------------------------------------------------------
## 1. Import packages
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from subroutines.warping_functions import *
from subroutines.time_frequency_analysis_functions import *

'''
Select the environment where you want to run the code Python Console or Terminal
To activate select 1, to deactivate mark 0 
Both environment cannot be activated at the same time
If you are using the Python Console you will have to close the figures to continue running the code 

We recommend to run the code in the terminal, this way you can see all the results (figures) 
at the end of the process
'''

PythonConsole=0
Terminal=1

if Terminal:
    matplotlib.use("TkAgg")

if PythonConsole == Terminal:
    raise ValueError ('Both environment cannot be activated/deactivated at the same time')



#--------------------------------------------------------------------------------------
## 2. Load simulated signal
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

#--------------------------------------------------------------------------------------
## 3. Plot time series

# The first sample of s_t_dec corresponds to time r/c1
# Make the signal shorter, no need to keep samples with zeros
N_ok = 150
s_ok = s_t_dec[0, 0:N_ok]

# Corresponding time and frequency axis
time = np.arange(1, N_ok + 1) / fs

print('The code has loaded a signal propagated in a Pekeris waveguide')
print('and shows the time series')

# Now, let's have a look at it
plt.figure()
plt.plot(time[0, :], s_ok[:])
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude (A.U.)')
plt.title('Signal in the time domain')
plt.grid()

if PythonConsole:
    print('Close the figure to continue')
    plt.show(block=True)
    # plt.ion()
    # input('Press ENTER to continue')

if Terminal:
    plt.get_current_fig_manager().window.wm_geometry("600x400+0+0")
    plt.show(block=False)
    input('Press ENTER to continue warping')


#--------------------------------------------------------------------------------------
## 4. Warping

# The warped signal will be s_w
s_w, fs_w = warp_temp_exa(s_ok, fs, r, c1)
M = len(s_w)
time_w = np.arange(0, M) / fs_w  ## time axis of the warped signal

print('\n' * 30)
print('This is the time series of the warped signal')
print('It is a stretched version on the original signal')


plt.figure()
plt.plot(time_w[0, :], s_w[:, 0])
plt.xlabel('Warped time (sec)')
plt.ylabel('Amplitude (A.U.)')
plt.title('Warped signal')
plt.grid()

if PythonConsole:
    print('Close the figure to continue and see the spectrogram')
    plt.show(block=True)

if Terminal:
    plt.get_current_fig_manager().window.wm_geometry("600x400+0+800")
    plt.show(block=False)
    input('Press ENTER to continue and see the spectrogram')


#--------------------------------------------------------------------------------------

## 5. Spectrograms
# Original signal

# STFT computation
NFFT = 1024
N_window = 31  # you need a short window to see the modes

b = np.arange(1, N_ok + 1)
d = np.hamming(N_window)
tfr = tfrstft(s_ok, b, NFFT, d)
spectro = abs(tfr) ** 2

# Figure
freq = (np.arange(0, NFFT)) * fs / NFFT
plt.figure()
plt.subplot(121)
plt.pcolormesh(time[0, :], freq[0, :], spectro)
plt.ylim([0, fs / 2])
plt.xlim([0, 0.5])  ### Adjust this to see better
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Original signal')

N_window_w = 301  ### You need a long window to see the warped modes
wind = np.hamming(N_window_w)
wind = wind / np.linalg.norm(wind)
t = np.arange(1, M + 1)
tfr_w = tfrstft(s_w, t, NFFT, wind)
spectro_w = abs(tfr_w) ** 2

# Frequency axis of the warped signal
freq_w = np.arange(0, NFFT) * fs / NFFT

# Figure
plt.subplot(122)
plt.pcolormesh(time_w[0, :], freq_w[0, :], spectro_w)
plt.ylim([0, 40])
plt.xlabel('Warped time (sec)')
plt.ylabel('Corresponding warped frequency (Hz)')
plt.title('Warped signal')

print('\n' * 30)
print('Here are the spectrograms')
print('Note that window length is highly different for')
print('the spectrogram of the original signal and the ')
print('spectrogram of the warped signal (look at codes and comments)')

if PythonConsole:
    print('Close the figure to proceed with inverse warping')
    plt.show(block=True)
    #plt.ion()

if Terminal:
    plt.get_current_fig_manager().window.wm_geometry("700x400+800+0")
    plt.show(block=False)
    input('Press ENTER to proceed with inverse warping')



# Let's unwarp the signal
s_r = iwarp_temp_exa(s_w, fs_w, r, c1, fs, N_ok)

plt.figure()
plt.plot(time[0, :], s_ok[:])
plt.plot(time[0, :], s_r[:, 0], 'or', fillstyle='none')
plt.grid()
plt.xlabel('Time (sec')
plt.title('blue: original ; red: after warping + unwarping')
print('\n' * 30)
print('The blue signal is the original signal (same as the first figure)')
print('The red dots is the signal after warping and inverse warping')
print('Notice how they perfectly fit')


if PythonConsole:
    print('Close the figure to continue and exit the code')
    plt.show(block=True)
    #plt.ion()

if Terminal:
    plt.get_current_fig_manager().window.wm_geometry("600x400+800+800")
    plt.show(block=False)
    input('Press ENTER to continue and exit the code')

print(' ')
print('END')
