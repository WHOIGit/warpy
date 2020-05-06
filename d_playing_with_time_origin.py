# Warping tutorial
## D_playing_with_time_origin

#### May 2020
#### Eva Chamorro Garrido

## Import packages
import os
import matplotlib
matplotlib.use('TkAgg')
#% matplotlib inline ## activate in jupyter notebook
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
import scipy.io as sio
from warping_functions import *
from time_frequency_analysis_functions import *
from roipoly import RoiPoly
from scipy.signal import hilbert

## Load simulated signal

data = sio.loadmat(os.getcwd() + '/sig_pek_for_warp.mat')

s_t = data['s_t']
fs = data['fs']
s_t_dec = data['s_t_dec']
r = data['r']
c1 = data['c1']

'''
    s_t: propagated modes in a Pekeris waveguide with parameters
    c1, c2, rho1, rho2: sound speed / density
        D: depth
        r: range
        zs, zr: source/receiver depth
    s_t_dec: same than s_t, except that time origin has been set for warping
    fs: sampling frequency


     NB: one can run optional_create_simulated_signal.m to generate another
     simulated signal
'''
xlim_plots = [6.5, 7.5]
N = len(s_t[0, :])
NFFT = 2048  # FFT size
N_window = 31  # sliding window size (need an odd number)

redo_dt = 0

while redo_dt == 0:
    a = np.arange(1, N + 1)
    a = a[np.newaxis, :]
    d = np.hamming(N_window)
    d = d[:, np.newaxis]
    tfr = tfrstft(s_t, a, NFFT, d)
    spectro = abs(tfr) ** 2

    time = np.arange(0, N) / fs
    freq = np.arange(0, NFFT) * fs / NFFT

    print('This is the spectrogram of a signal propagated in a Pekeris waveguide')
    print('In this code, you will choose the time origin for warping')
    print('Click once on the spectrogram at the position where you want to define the time origin')

    #% matplotlib qt ## activate in jupyter notebook
    plt.figure()
    plt.pcolormesh(time[0, :], freq[0, :], spectro)
    plt.ylim([0, fs / 2])
    plt.xlim(xlim_plots)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.show(block=False)

    interactive(True)
    t_dec = plt.ginput(1)
    t_dec = t_dec[0]
    t_dec = t_dec[0]

    ## Shorten the signal, make it start at the chosen time origin, and warp it
    time_t_dec = np.abs(time - t_dec)
    ind0 = np.where(time_t_dec == np.min(time_t_dec))
    ind0 = ind0[1]
    ind0 = ind0[0]

    N_ok = 150
    s_ok = s_t[:, ind0:ind0 + N_ok]

    # Corresponding time and frequency axis
    time_ok = np.arange(0, N_ok) / fs

    # The warped signal will be s_w
    [s_w, fs_w] = warp_temp_exa(s_ok, fs, r, c1)
    M = len(s_w)

    ## Plot spectrograms

    ### Original signal
    N_window = 31  # you need a short window to see the modes

    a = np.arange(1, N_ok + 1)
    a = a[np.newaxis, :]
    d = np.hamming(N_window)
    d = d[:, np.newaxis]
    tfr = tfrstft(s_ok, a, NFFT, d)
    spectro = abs(tfr) ** 2

    ### Warped signal
    N_window_w = 301  # You need a long window to see the warped modes
    wind = np.hamming(N_window_w)
    wind = wind / np.linalg.norm(wind)
    wind = wind[:, np.newaxis]
    b = np.arange(1, M + 1)
    b = b[np.newaxis, :]
    tfr_w = tfrstft(s_w, b, NFFT, wind)
    spectro_w = abs(tfr_w) ** 2

    # Time and frequency axis of the warped signal
    time_w = np.arange(0, (M) / fs_w, 1 / fs_w)
    freq_w = np.arange(0, fs_w - fs_w / NFFT + fs_w / NFFT, fs_w / NFFT)

    # Figure
    #% matplotlib inline ## activate in jupyter notebook
    plt.figure(figsize=(10.0, 8.0))
    plt.subplot(121)
    plt.pcolormesh(time_ok[0, :], freq[0, :], spectro)
    plt.ylim([0, fs / 2])
    # plt.xlim([0, 0.5])  ### Adjust this to see better
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Original signal with chosen time origin')

    # Figure
    plt.subplot(122)
    plt.pcolormesh(time_w, freq_w, spectro_w)
    plt.ylim([0, 40])  ### Adjust this to see better
    plt.xlabel('Warped time (sec)')
    plt.ylabel('Corresponding warped frequency (Hz)')
    plt.title('Corresponding warped signal')
    plt.show()

    print('The left panel shows the spectrogram of the original ')
    print('signal with the chosen time origin')
    print('The right panel shows the corresponding warped signal')
    print('Type 0 if you want to redo the time origin selection')
    redo_dt = input('or type 1 if you want to proceed with modal filtering ')

    if redo_dt == '' or redo_dt == '1':
        redo_dt = 1

    elif redo_dt == '0':
        redo_dt = int(redo_dt)

## selection the spectro_w part to mask

spectro_w_1 = spectro_w[0:400, :]
Nmode = 4
modes = np.zeros((N_ok, Nmode))
tm = np.zeros((NFFT, Nmode))

#% matplotlib qt ## activate in jupyter notebook

# Figure
plt.figure(figsize=(10, 10))
plt.imshow(spectro_w_1)
plt.ylim([0, 400])
plt.xlabel('Warped time (sec)')
plt.ylabel('Corresponding warped frequency (Hz)')
plt.title('Filter Mode 1')
plt.show(block=False)

## Filtering
# To make it easier, filtering will be done by hand using the roipoly tool.
# See python help for more information

print('Now try to filter the 4 modes');
print('Create the four masks sequentially, starting with mode 1, then 2, etc.')
print('(if needed, go back to c_warping_and_filtering.m for mask creation)')
input('Press ENTER to filter mode 1')

plt.close('all')

color = ['r', 'b', 'g', 'yellow']

for i in range(Nmode):
    plt.figure(figsize=(10, 10))
    plt.imshow(spectro_w_1)
    plt.ylim([0, 400])
    plt.xlabel('Warped time (sec)')
    plt.ylabel('Corresponding warped frequency (Hz)')
    plt.title('Filter Mode ' + str(i + 1))
    plt.show(block=False)

    # Let user draw ROI
    interactive(True)
    roi1 = RoiPoly(color=color[i])

    if i <= 2:
        input('Press ENTER to filter mode' + str(i + 2))
    else:
        input('Press ENTER to continue')

    ## Filtering

    # create the mask of roi1
    mask = roi1.get_mask(spectro_w_1)
    masque_1 = np.double(mask)

    # add the part masked to the total sprectogram array
    masque_2 = np.zeros_like(spectro_w[400:, :])
    masque = np.concatenate((masque_1, masque_2), axis=0)

    mode_rtf_warp = masque * tfr_w
    norm = 1 / NFFT / np.max(wind)
    mode_temp_warp = np.real(np.sum(mode_rtf_warp, axis=0)) * norm * 2
    mode = iwarp_temp_exa(mode_temp_warp, fs_w, r, c1, fs, N_ok)
    modes[:, i] = mode[:, 0]

    ## Verification

    a = hilbert(modes[:, i])
    a = a[:, np.newaxis]
    b = np.arange(1, N_ok + 1)
    b = b[np.newaxis, :]
    d = np.hamming(N_window)
    d = d[:, np.newaxis]
    mode_stft = tfrstft(a, b, NFFT, d)
    mode_spectro = abs(mode_stft) ** 2
    tm_a = tm[:, i]
    tm_a = tm_a[:, np.newaxis]
    tm_a, D2 = momftfr(mode_spectro, 0, N_ok, time_ok)
    tm[:, i] = tm_a[:, 0]

#% matplotlib inline ## activate in jupyter notebook

plt.figure()
plt.pcolormesh(time_ok[0, :], freq[0, :], spectro)
plt.ylim([0, fs / 2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram and estimated dispersion curve')
plt.plot(tm[:, 0], freq[0, :], 'r')
plt.plot(tm[:, 1], freq[0, :], 'r')
plt.plot(tm[:, 2], freq[0, :], 'r')
plt.plot(tm[:, 3], freq[0, :], 'r')
plt.show()

print('The red lines are the estimated dispersion curves.')
input('Press ENTER to continue')

## Last verification

data = sio.loadmat(os.getcwd() + '/sig_pek_and_modes_for_warp.mat')
r = data['r']
vg = data['vg']
c1 = data['c1']
f_vg = data['f_vg']

### creation of the theoretical dispersion curves
tm_theo = r / vg - r / c1  ### range over group_speed minus correction for time origin

plt.figure()
plt.pcolormesh(time_ok[0, :], freq[0, :], spectro)
plt.ylim([0, fs / 2])
plt.xlim([0, 0.74])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram and estimated dispersion curve')
plt.plot(tm_theo[0, :], f_vg[0, :], 'black')
plt.plot(tm_theo[1, :], f_vg[0, :], 'black')
plt.plot(tm_theo[2, :], f_vg[0, :], 'black')
plt.plot(tm_theo[3, :], f_vg[0, :], 'black')
plt.plot(tm[:, 0], freq[0, :], 'r')
plt.plot(tm[:, 1], freq[0, :], 'r')
plt.plot(tm[:, 2], freq[0, :], 'r')
plt.plot(tm[:, 3], freq[0, :], 'r')
plt.show()

print('This is the same figure than before, except that the true dispersion curves are now in black.')
print('How well did you do?')
print(' ')
print('Recall that you have to restrict interpretation of the dispersion curve (red line)')
print('to only the frequency band where it is relevant. The black and red lines will not match')
print('entirely for the length of the red line')
print(' ')

print('For practical applications, you will need to restrict dispersion curves to a frequency band')
print('where they are ok. This will be covered in the script g_filtering_multiple_modes_for_loc.')
print(' ')
print('Try to rerun the code and change the time origin to overly late/early value and see what it does.')
print(' ')
print('END')