## Warping tutorial
###Julien Bonnel-Eva Chamorro
##April 2020

import numpy as np
#%matplotlib qt # Activate for python in jupyter notebook
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from scipy.fftpack import fft
from matplotlib import interactive
from warping_functions import *
from tfrstft import tfrstft
from momftfr import momftfr
from roipoly import RoiPoly, MultiRoi
from scipy.signal import hilbert

## Load simulated signal
data = sio.loadmat(os.getcwd() + '/sig_pek_for_warp.mat')

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
#Load variables
s_t = data['s_t']
fs = data['fs']
s_t_dec = data['s_t_dec']
r = data['r']
c1 = data['c1']


## Process signal
# The first sample of s_t_dec corresponds to time r/c1

# Make the signal shorter, no need to keep samples with zeros
N_ok=150
s_ok=s_t_dec[:,0:N_ok]

# Corresponding time and frequency axis
time=np.arange(0,N_ok)/fs

# The warped signal will be s_w
[s_w, fs_w]=warp_temp_exa(s_ok,fs,r,c1)
M=len(s_w)


# Time frequency representations
# Original signal
# STFT computation
NFFT=1024
N_window=31  ### you need a short window to see the modes
b=np.arange(1,N_ok+1)
b=b[np.newaxis,:]
tfr,t,f=tfrstft(s_ok,b,NFFT,N_window)
spectro=abs(tfr)**2

# Figure
freq=(np.arange(0,NFFT))*fs/NFFT
plt.figure(figsize=(7,5))
plt.pcolormesh(time[0, :], freq[0, :], spectro)

plt.ylim([0, fs/2])
plt.xlim([0, 0.5])  ### Adjust this to see better
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Original signal')
plt.show()

print('This is the spectrogram of the received signal')
print('We will now warp it')
input('Press ENTER to continue')


# Warped signal
# STFT computation

N_window_w=301 ### You need a long window to see the warped modes
wind=np.hamming(N_window_w)
wind=wind/np.linalg.norm(wind)
t=np.arange(1,M+1)
t=t[np.newaxis,:]
tfr_w,t_w,f_w=tfrstft(s_w,t,NFFT,N_window_w)
spectro_w=abs(tfr_w)**2

# Time and Frequency axis of the warped signal
time_w=np.arange(0,(M)/fs_w,1/fs_w)
freq_w=np.arange(0,NFFT)*fs/NFFT

# Figure
plt.figure(figsize=(7,5))
#plt.pcolormesh(time_w, freq_w[0, :], spectro_w)
plt.imshow(spectro_w)
#plt.ylim([0,40])
#plt.ylim([0,200])
#plt.xlim([0,450])
plt.xlabel('Warped time (sec)')
plt.ylabel('Corresponding warped frequency (Hz)')
plt.title('Warped signal')
plt.show(block=False)

## Filtering
# To make it easier, filtering will be done by hand using the roipoly tool.
# See Python roipoly PypI for more information
print('We will now filter a single mode by creating a time-frequency mask.')
print('To do so, click several times on the spectrogram to define the region you want to filter.')
print('As a last click, you must double-click and and Matlab will close the polygon')
print('You can then click on the vertices and drag them to adjust the polygon shape, or click within the polygon to slide it.')
print('Once you are ok with the mask shape, right click and select "Create Mask".')
print('Look at Fig. 11 in the paper for a mask shape suggestion')
print('(you can enlarge the figure or make it full screen before creating the mask)')

#Let user draw ROI
interactive(True)
roi1 = RoiPoly(color='r')

# Create ROI masks
mask=roi1.get_mask(spectro_w)
masque=np.double(mask)

#in case draw a ROI is not working for run the code i have upload some real data
#data = sio.loadmat(os.getcwd()+ '/BW.mat')
#BW=data['BW']
#masque=np.double(BW)

#Note that the mask is applied on the STFT (not on the spectrogram)
mode_rtf_warp=masque*tfr_w
norm=1/NFFT/np.max(wind)
mode_temp_warp=np.real(np.sum(mode_rtf_warp,axis=0))*norm*2
# The modes are actually real quantities, so that they have negative frequencies.
# Because we do not filter negative frequencies, we need to multiply by 2
# the positive frequencies we have filter in order to recover the correct
# modal energy
mode=iwarp_temp_exa(mode_temp_warp,fs_w,r,c1,fs,N_ok)


## Verification
# you can estimate the dispersion curve by computing the
# frequency moment of the filtered mode TFR

a=hilbert(mode)
b=np.arange(1,N_ok+1)
b=b[np.newaxis,:]
mode_stft,t,f=tfrstft(a,b,NFFT,N_window)
mode_spectro=abs(mode_stft)**2
tm,D2=momftfr(mode_spectro,0,N_ok,time)


plt.figure()
plt.pcolormesh(time[0, :], freq[0, :], spectro)
plt.ylim([0,fs/2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram and estimated dispersion curve')


plt.plot(tm[:,0],freq[0, :],'r')


print('The red line is the estimated dispersion curve.')
print('You have to restrict it to a frequency band where it is relevant: the red line')
print('is supposed to follow the mode for frequencies where the mode has a high amplitude,')
print('but it can look noisy at frequencies where the mode has a low amplitude (e.g. at frequencies below cutoff) ')
print('...')
print('If the result is not satisfying, you must create a new filtering mask')
print('If the result is ok, you can try another mode!')
print('In any case, let us look at your result vs the true modes')
input('Press ENTER to continue...')

## Last verification
data = sio.loadmat(os.getcwd()+ '/sig_pek_and_modes_for_warp.mat')
r=data['r']
vg=data['vg']
c1=data['c1']
f_vg=data['f_vg']

### creation of the theoretical dispersion curves

tm_theo=r/vg-r/c1; ### range over group_speed minus correction for time origin

plt.figure()
plt.pcolormesh(time[0, :], freq[0, :], spectro)
plt.ylim([0,fs/2])
plt.xlim([0,0.74])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram and estimated dispersion curve')
plt.plot(tm_theo[0,:], f_vg[0,:], 'black')
plt.plot(tm_theo[1,:], f_vg[0,:], 'black')
plt.plot(tm_theo[2,:], f_vg[0,:], 'black')
plt.plot(tm_theo[3,:], f_vg[0,:], 'black')
plt.plot(tm[:,0],freq[0, :],'red')

print('This is the same figure than before, except that the true dispersion curves are now in black.')
print('How well did you do?')
print(' ')
print('Recall that you have to restrict interpretation of the dispersion curve (red line)')
print('to only the frequency band where it is relevant. The black and red lines will not match')
print('entirely for the length of the red line')
print(' ')
print('END')