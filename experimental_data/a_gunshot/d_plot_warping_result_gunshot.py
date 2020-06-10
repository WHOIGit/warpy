## Warping tutorial
#### d_plot_warping_result

##### May 2020
###### Eva Chamorro - Daniel Zitterbart - Julien Bonnel

## 1. Import packages

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from datetime import date
import warnings
warnings.filterwarnings('ignore')


## 2. Load data


#### Option to plot spectrogram in dB vs linear scale
plot_in_dB=1  ### 1 to plot in dB, or 0 to plot in linear scale

## Load data
print('This code is to plot warping results')
print('Select the .mat file with the results you want to plot')
print('(this .mat file has been created by b_filtering.m)')
input('Press ENTER to continue')

today = date.today()
#dat = sio.loadmat(os.getcwd()+ '/modes_' + str(today)+ '.mat')
dat = sio.loadmat(os.getcwd()+ '/gunshot_modes_2020-06-08.mat')

data=dat['data']
fmax_plot=dat['fmax_plot']
freq_data=dat['freq_data']
freq=dat['freq']
freq_w=dat['freq_w']
time_ok=dat['time_ok']
time_w=dat['time_w']
spectro=dat['spectro']
spectro_w=dat['spectro_w']
fs=dat['fs']
mode_pts=dat['mode_pts']
mode_pts=mode_pts[0]
Nmodes=dat['Nmode']

## 3. Spectroprint(' ')
print('\n' * 20)
print('The left panel shows the spectrogram of the original signal and the estimated dispersion curves.')
print('The right panel shows the spectrogram of the warped signal and the TF masks that were used to filter the modes.')
print(' ')
print('Close the figure to exit the code')


plt.figure()
plt.subplot(121)
if plot_in_dB == 1:
    s = 10 * np.log10(spectro)
    plt.imshow(s, extent=[time_ok[0, 0], time_ok[0, -1], freq[0, 0], freq[0, -1]], aspect='auto', origin='low')
else:
    s = spectro
    plt.imshow(s, extent=[time_ok[0, 0], time_ok[0, -1], freq[0, 0], freq[0, -1]], aspect='auto', origin='low')

plt.ylim([0, int(fs) / 2])
plt.xlabel('Time [sec]')
plt.ylabel('Frequency [Hz]')
plt.title('Spectrogram')

c = ['black', 'blue', 'green', 'red']
for i in range (int(Nmodes)):
    plt.plot(data[:, i], freq_data[0, :], color=c[i])


plt.subplot(122)
if plot_in_dB == 1:
    s_w = 10 * np.log10(spectro_w)
    plt.imshow(s_w, extent=[time_w[0, 0], time_w[0, -1], freq_w[0, 0], freq_w[0, -1]], aspect='auto', origin='low')

else:
    s_w = spectro_w
    plt.imshow(s_w, extent=[time_w[0, 0], time_w[0, -1], freq_w[0, 0], freq_w[0, -1]], aspect='auto', origin='low')

plt.ylim([0, fmax_plot])  ### Adjust this to see better
plt.xlabel('Warped time (sec)')
plt.ylabel('Corresponding warped frequency (Hz)')
plt.title('Corresponding warped signal')

## create the mask points to plot on the warped signal


for i in range(len(data[0, :])):
    pts_x1 = []
    pts_y1 = []

    point = mode_pts[i]
    add = point[0, :]
    add = add[np.newaxis, :]
    point = np.append(point, add, axis=0)
    point = np.round(point)

    for j in range(len(point)):
        pts_x1 = np.append(pts_x1, time_w[0, int(point[j, 0])])
        pts_y1 = np.append(pts_y1, freq_w[0, int(point[j, 1])])

    plt.plot(pts_x1, pts_y1, color=c[i])

plt.show(block='True')

print('')
print('END')
