
## Warping tutorial
###Julien Bonnel-Eva Chamorro
##April 2020

##THIS CODE IS TO LOCALIZE FM SOURCES
#matplotlib inline

#import packages
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from spectrum import *
import scipy.io
import os


## Load simulated signal
data = scipy.io.loadmat(os.getcwd()+ '/sig_pek_for_warp.mat')
#print(data.keys())
'''
# s_t: propagated modes in a Pekeris waveguide with parameters
#      c1, c2, rho1, rho2: sound speed / density
#      D: depth
#      r: range
#      zs, zr: source/receiver depth
# s_t_dec: same than s_t, except that time origin has been set for warping
# fs: sampling frequency
'''
s_t=data['s_t']
fs=data['fs']

'''
### NB: one can run optional_create_simulated_signal.m to generate another
### simulated signal
### IF YOU CHANGE RANGE, you must change the following variable which
### defines the bounds on the x-axis for all the plots
'''
xlim_plots=[6.5,7.5]

## Short Fourier Transform computation

def tfrstft(x, t, N, N_window):
    h = np.hamming(N_window);
    h = h[:, np.newaxis];

    (xrow, xcol) = np.shape(x);

    ## Lines added by Bonnel
    if ((xcol != 1)):
        x = np.transpose(x);
        (xrow, xcol) = np.shape(x);
    ## end of Bonnel's addition

    hlength = np.floor(NFFT / 4);
    hlength = hlength + 1 - np.remainder(hlength, 2);
    trace = 0;
    (trow, tcol) = np.shape(t);
    (hrow, hcol) = np.shape(h);
    Lh = (hrow - 1) / 2;
    h = h / np.linalg.norm(h);
    tfr_2 = np.zeros((NFFT, tcol));

    for icol in range(tcol):
        ti = t[0, icol];
        tau = np.arange(-np.min([np.round(N / 2) - 1, Lh, ti - 1]), np.min([np.round(N / 2) - 1, Lh, xrow - ti]) + 1);
        indices = np.remainder(N + tau, N) + 1;
        indices = indices.astype(int);
        a = np.array(Lh + 1 + tau, dtype='int');
        b = np.array(ti + tau, dtype='int');
        c = x[b - 1, :] * np.conj(h[a - 1]);
        tfr_2[indices - 1, icol] = c[:, 0];

    tfr = fft(tfr_2, axis=0)

    return tfr


## Compute time-frequency representation

# Signal length
N=np.size(s_t);

# Short Time Fourier Transform parameters
vec_t=np.arange(0,N,1); vec_t = vec_t[np.newaxis,:]; # time samples where the STFT will be computed (i.e. positions of the sliding window)

NFFT=2048; # FFT size
N_window=31; # sliding window size (need an odd number)


#STFT computation

tfr=tfrstft(s_t,vec_t,NFFT,N_window);

# Spectrogram ~ modulus STFT
spectro=abs(tfr)**2;

# Time and frequency axis
time=np.arange(0,N)/fs;
freq=np.arange(0,NFFT)*fs/NFFT;



plt.figure(figsize=(9.0,7.0))
plt.plot(time[0,:], s_t[0,:])
plt.xlim(xlim_plots)
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude (A.U.)')
plt.title('Signal in the time domain')
plt.grid()

print('The code has loaded a signal propagated in a Pekeris waveguide')
print('and shows the time series')
print('Press any key to continue and see the spectrogram')

plt.pause(0.5)

plt.close('all')

plt.figure(figsize=(9.0,7.0))


plt.pcolormesh(time[0,:],freq[0,:],spectro)
plt.xlim(xlim_plots)
plt.ylim([0,fs/2])

plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram')


print('This is the spectrogram, computed with a sliding window ')
print('of length Nw=31 samples (or 0.15 s , since the sampling frequency is 200 Hz')
print('As a reminder, the spectrogram is computed by dividing the original signal into')
print('segments of length Nw. Another important parameter for spectrogram computation is')
print('overlap. In this tutorial, the overlap between segment is Nw-1 samples, which is ')
print('the maximum overlap that one can obtain : the percentage of overlap is 100*(Nw-1)/Nw')
print('Such a high overlap will be handy for modal filtering')
print('Press any key to continue')
plt.pause(0.5)


print('Now you can modify the window length')
print('Remember to give it in number of samples')
print('Number between 11 and 71 are reasonable tries')
print('(for coding reason, please use odd numbers only)')


##This part is not working well only works one time
# 'Continue' is not working I think is a code style parameter of PyCharm
# I have to investigate because in Jupyter is working well

while ((N_window != 999)):
    print('Input window length and press enter for spectrogram computation : ')
    N_window = int(input('(window length 0 to exit the code) '));

    if ((N_window == 0) or (N_window == 999)):
        print('  ')
        print('END')
        break

    if ((N_window != 999)):
        # STFT computation
        tfr = tfrstft(s_t, vec_t, NFFT, N_window);
        # Spectrogram ~ modulus STFT
        spectro = abs(tfr) ** 2;
        plt.figure(figsize=(9.0,7.0))
        plt.pcolormesh(time[0, :], freq[0, :], spectro)
        plt.ylim([0, fs / 2])
        plt.xlim(xlim_plots)
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram N_window: '+ str(N_window))
        plt.show()
        continue





