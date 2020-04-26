## Warping tutorial
###Julien Bonnel-Eva Chamorro
##April 2020

#import packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from tfrstft import tfrstft
from warping_functions import *  #I have define all the functions toguether in the python file warping_functions.py



## Load simulated signal
data = sio.loadmat(os.getcwd()+ '/sig_pek_for_warp.mat')
#print(data.keys())


s_t_dec=data['s_t_dec']
fs=data['fs']
r=data['r']
c1=data['c1']
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

##Plot time series
# The first sample of s_t_dec corresponds to time r/c1

# Make the signal shorter, no need to keep samples with zeros
N_ok=150;
s_ok=s_t_dec[0,0:N_ok]
s_ok=s_ok[np.newaxis,:]

# Corresponding time and frequency axis
time=np.arange(0,N_ok)/fs

# Now, let's have a look at it
plt.figure(figsize=(7,5))
plt.plot(time[0,:],s_ok[0,:])
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude (A.U.)')
plt.grid()
plt.title('Signal in the time domain')
plt.show()

print('The code has loaded a signal propagated in a Pekeris waveguide')
print('and shows the time series')
print('Press any key to proceed with warping')
input('Press ENTER to continue...')

## Warping

# The warped signal will be s_w
[s_w, fs_w]=warp_temp_exa(s_ok,fs,r,c1)
M=len(s_w)
time_w=np.arange(0,M)/fs_w ## time axis of the warped signal


plt.figure(figsize=(7,5))
plt.plot(time_w[0,:],s_w[:,0])
plt.grid()
plt.xlabel('Warped time (sec)')
plt.ylabel('Amplitude (A.U.)')
plt.title('Warped signal')
plt.show()

print('This is the time series of the warped signal')
print('It is a stretched version on the original signal')
print('Press any key to proceed with spectrograms')
input('Press ENTER to continue...')

###Spectrograms
###Original signal
# STFT computation
NFFT=1024
N_window=31 # you need a short window to see the modes

b=np.arange(1,N_ok+1)
b=b[np.newaxis,:]
tfr=tfrstft(s_ok,b,NFFT,N_window)
spectro=abs(tfr)**2

# Figure
freq=(np.arange(0,NFFT))*fs/NFFT
plt.figure(figsize=(10,7))
plt.subplot(121)
plt.pcolormesh(time[0, :], freq[0, :], spectro)
plt.ylim([0, fs/2])
plt.xlim([0, 0.5])  ### Adjust this to see better
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Original signal')
plt.show()

N_window_w=301; # You need a long window to see the warped modes
wind=np.hamming(N_window_w);
wind=wind/np.linalg.norm(wind);
t=np.arange(1,M+1)
t=t[np.newaxis,:]
tfr_w=tfrstft(s_w,t,NFFT,N_window_w);

spectro_w=abs(tfr_w)**2;

## Frequency axis of the warped signal
freq_w=np.arange(0,NFFT)*fs/NFFT;
# Figure
plt.subplot(122)
plt.pcolormesh(time_w[0, :], freq_w[0, :], spectro_w)
plt.ylim([0,40])
plt.xlabel('Warped time (sec)')
plt.ylabel('Corresponding warped frequency (Hz)')
plt.title('Warped signal')
plt.show()


print('Here are the spectrograms')
print('Note that window length is highly different for')
print('the spectrogram of the original signal and the ')
print('spectrogram of the warped signal (look at codes and comments)')
print('Press any key to proceed with inverse warping')
input('Press ENTER to continue...')

## Let's unwarp the signal
s_r=iwarp_temp_exa(s_w,fs_w,r,c1,fs,N_ok);


plt.figure(figsize=(7,5))
plt.plot(time[0,:],s_ok[0,:])
plt.plot(time[0,:],s_r[:,0],'or',fillstyle='none')
plt.grid ()
plt.xlabel('Time (sec')
plt.title('blue: original ; red: after warping + unwarping')
plt.show()

print('The blue signal is the original signal (same as the first figure)')
print('The red dots is the signal after warping and inverse warping')
print('Notice how they perfectly fit')
print(' ')
print('END')


