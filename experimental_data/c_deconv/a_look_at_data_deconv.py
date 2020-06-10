## Warping tutorial
#### a_look_at_data, deconv

##### April 2020
###### Eva Chamorro - Daniel Zitterbart - Julien Bonnel

## 1. Import packages

import os
os.chdir("/Users/evachamorro/PycharmProjects/WHOI")
#**Put here the directory where you have the file with your function**
import wave
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.io.wavfile as siw
import scipy.signal as sig
from ipywidgets import interact_manual
from warping_functions import *
from time_frequency_analysis_functions import *
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.io.wavfile import write
import warnings
warnings.filterwarnings('ignore')

os.chdir('/Users/evachamorro/PycharmProjects/WHOI/c_css_deconv')
##**Put here the directory where you were working**

## 2. Intro

print('This last script illustrates source deconvolution of a Combustive Sound Source (CSS) signal')
print('Because of requirements associated with source deconvolution, this script is')
print('less interactive than the others. The code is nonetheless heavily commented')
print('so that an interested user could adapt it to his own data')
print(' ')


path=os.getcwd() + '/css_received_signal.wav'
fs, s = siw.read(path)
f = wave.open(os.path.join(path,path),'rb')

N=len(s)

path=os.getcwd() + '/css_source_signal.wav'
fs_source, s_source = siw.read(path)
f_s = wave.open(os.path.join(path,path),'rb')

Nsource=len(s_source)

print('Received signal read from css_received_signal.wav')
print('CSS source signal read from css_source_signal.wav')



## 3. First plots

print('The plot shows the source signal (left) and the signal received after propagation over several kilometers (right)')
print('Notice that the time scales on the x-axis are different')
print('The propagated signal is longer than the source signal, which is a good hint of dispersion')
print('Close the figure to plot spectrograms')
print('')
print('')

plt.figure()
plt.subplot(121)

plt.plot(np.arange(1,Nsource+1)/fs_source, s_source)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')
plt.title('CSS source signal recorded 1 m from the source')
plt.grid()

plt.subplot(122)
plt.plot(np.arange(1,N+1)/fs, s)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')
plt.title('Received signal after propagation over several km')
plt.grid()
plt.show(block='True')


## 4. Spectrograms

NFFT=1024
Nw=303
Nw_source=81  ### the source is impulsive, we need a short window

t=np.arange(0,N)/fs
t_source=np.arange(0,Nsource)/fs
freq=np.arange(0,NFFT)*fs/NFFT
freq_source=np.arange(0,NFFT)*fs_source/NFFT

a=s[:,np.newaxis]
b=np.arange(1,N+1)
b = b[np.newaxis, :]
h=np.hamming(Nw)
h = h[:, np.newaxis]
rtf=tfrstft(a, b, NFFT, h)

a=s_source[:,np.newaxis]
b=np.arange(1,Nsource+1)
b = b[np.newaxis, :]
h=np.hamming(Nw_source)
h = h[:, np.newaxis]
rtf_source=tfrstft(a, b, NFFT, h)

plt.figure()
plt.subplot(121)
plot=10*np.log10(abs(rtf_source)**2)
plt.imshow(plot,extent=[t_source[0], t_source[-1], freq_source[0], freq_source[-1]], aspect='auto', origin='low')

plt.ylim([0,fs_source/2])

plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('CSS source signal recorded 1 m from the source')

plt.subplot(122)
plot=10*np.log10(abs(rtf)**2)
plt.imshow(plot,extent=[t[0], t[-1], freq[0], freq[-1]], aspect='auto', origin='low')
plt.ylim([0,fs/2])

plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title({'Received signal after propagation over several km'})
plt.show(block='True')

print('As previously, the time scales on the x-axis are different')
print('Moreover, notice that the frequency scales on the y-axis are different: this is because the two signals')
print('have a different sampling frequency. We will need to modify that before source deconvolution')
print('Close the figure to change the sampling frequency of the source signal')


## 5. Set a common frequency

s_source_ok= signal.resample_poly(s_source,round(fs*1000),fs_source*1000) ### easy trick to obtain the same sampling frequency
N_source_ok=len(s_source_ok)


plt.figure()

plt.plot(np.arange(0,Nsource)/fs_source, s_source)

plt.plot(np.arange(0,N_source_ok)/fs, s_source_ok,  'or', fillstyle='none')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')
#legend('Original source signal', 'Source signal after decimation')
plt.grid()
plt.show(block='True')


print('The source signal has a higher sampling frequency than the propagated signal')
print('The source signal must be decimated so that is has the same frequency than the propagated signal')
print('The plot shows the decimation result')
print('Close the figure to proceed with deconvolution')

## 6. Deconvolution

NFFT_deconv=max(N_source_ok, N)

sig_f=fft(s, NFFT_deconv)
source_f=fft(s_source_ok, NFFT_deconv)

eee=max(abs(source_f)**2)/100

sig_deconv_f=(sig_f*np.conj(source_f))/np.maximum(abs(source_f)**2,eee) ### deconvolution using Eq (26)
sig_deconv_t=ifft(sig_deconv_f)

#### reviewww make real signal ok???not sure

sig_deconv_t_t=np.real(sig_deconv_t)

Nd=NFFT_deconv
time_d=np.arange(0,Nd)/fs

a=sig_deconv_t_t[:,np.newaxis]
b=np.arange(1,Nd+1)
b = b[np.newaxis, :]
h=np.hamming(Nw)
h = h[:, np.newaxis]

rtf_deconv=tfrstft(a, b, NFFT, h)



plt.figure()
plot=10*np.log10(abs(rtf_deconv)**2)
plt.imshow(plot,extent=[time_d[0], time_d[-1], freq[0], freq[-1]], aspect='auto', origin='low')

plt.ylim([0,fs/2])
plt.xlim([0,0.5])

plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Received signal after propagation and deconvolution')
plt.show(block='True')

write('css_ready_to_warp.wav', fs, sig_deconv_t_t)

print('The plot shows the spectrogram of the propagated signal after source deconvolution')
print('The signal is now ready for warping. It is saved into css_ready_to_warp.wav')
print('Before warping, let us decimate the signal a bit')
print('Close the figure to proceed with decimation')

## 7. Decimation

Nsub=10
fs_ok=fs/Nsub
s_ok=signal.decimate(sig_deconv_t,Nsub)
N_ok=len(s_ok)

t_ok=np.arange(0,N_ok)/fs_ok
freq_ok=np.arange(0,NFFT)*fs_ok/NFFT


Nw_ok=51

a=s_ok[:,np.newaxis]
b=np.arange(1,N_ok+1)
b = b[np.newaxis, :]
h=np.hamming(Nw_ok)
h = h[:, np.newaxis]

rtf_ok=tfrstft(a, b, NFFT, h)

print('Here is the spectrogram of the signal after decimation by a factor of 10')
print('You can now go the the next script to filter the mode using warping')
print('The parameters to use will be')
print('   * decimation rate : 10')
print('   *  sliding window length : 51')
print('Close the figure to exit the code ')
print('  ')

plt.figure()
plot=10*np.log10(abs(rtf_ok)**2)

plt.imshow(plot,extent=[t_ok[0], t_ok[-1], freq_ok[0], freq_ok[-1]], aspect='auto', origin='low')

plt.ylim([0,fs_ok/2])
plt.xlim([0,0.5])

plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Received signal after propagation, deconvolution and decimation')
plt.show(block='True')



print('  ')
print('END')

