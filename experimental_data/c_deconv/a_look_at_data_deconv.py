# Warping tutorial
# a_look_at_data, deconv

# April 2020
# Eva Chamorro - Daniel Zitterbart - Julien Bonnel

#--------------------------------------------------------------------------------------
# 1. Import packages

import os
import sys
import wave
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as siw
sys.path.insert(0, os.path.dirname(os.getcwd())+'/subroutines')
#**Put here the directory where you have the file with your function**
from warping_functions import *
from time_frequency_analysis_functions import *
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.io.wavfile import write
import warnings
warnings.filterwarnings('ignore')

'''
Select the environment where you want to run the code (Python Console or Terminal).
To activate select 1, to deactivate select 0 
Both environment cannot be activated at the same time
If you are using the Python Console you will have to close the figures to continue running the code 

We recommend to run the code in the Terminal, this way you can see all the results (figures) you don't have 
to close the figures

'''

PythonConsole=1
Terminal=0

if Terminal:
    matplotlib.use("TkAgg")

if PythonConsole == Terminal:
    raise ValueError ('Both environment cannot be activated/deactivated at the same time')



#--------------------------------------------------------------------------------------
# 2. Intro
print('\n'*20)
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


#--------------------------------------------------------------------------------------
# 3. First plots
print('\n'*20)
print('The plot shows the source signal (left) and the signal received after propagation over several kilometers (right)')
print('Notice that the time scales on the x-axis are different')
print('The propagated signal is longer than the source signal, which is a good hint of dispersion')
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

if PythonConsole:
    print('Close the figure to to plot spectrograms')
    plt.show(block=True)
    # plt.ion()
    # input('Press ENTER to continue')

if Terminal:
    plt.get_current_fig_manager().window.wm_geometry("800x400+0+0")
    plt.show(block=False)
    input('Press ENTER to plot spectrograms')

#--------------------------------------------------------------------------------------
# 4. Spectrograms

NFFT=1024
Nw=303
Nw_source=81  ### the source is impulsive, we need a short window

t=np.arange(0,N)/fs
t_source=np.arange(0,Nsource)/fs
freq=np.arange(0,NFFT)*fs/NFFT
freq_source=np.arange(0,NFFT)*fs_source/NFFT


b=np.arange(1,N+1)
h=np.hamming(Nw)
rtf=tfrstft(s, b, NFFT, h)

b=np.arange(1,Nsource+1)
h=np.hamming(Nw_source)
rtf_source=tfrstft(s_source, b, NFFT, h)

plt.figure()
print('\n'*20)
print('As previously, the time scales on the x-axis are different')
print('Moreover, notice that the frequency scales on the y-axis are different: this is because the two signals')
print('have a different sampling frequency. We will need to modify that before source deconvolution')

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
if PythonConsole:
    print('Close the figure tto change the sampling frequency of the source signal')
    plt.show(block=True)
    # plt.ion()
    # input('Press ENTER to continue')

if Terminal:
    plt.get_current_fig_manager().window.wm_geometry("600x400+800+0")
    plt.show(block=False)
    input('Press ENTER to change the sampling frequency of the source signal')




#--------------------------------------------------------------------------------------
# 5. Set a common frequency

s_source_ok= signal.resample_poly(s_source,round(fs*1000),fs_source*1000) ### easy trick to obtain the same sampling frequency
N_source_ok=len(s_source_ok)


plt.figure()
print('\n'*20)
print('The source signal has a higher sampling frequency than the propagated signal')
print('The source signal must be decimated so that is has the same frequency than the propagated signal')
print('The plot shows the decimation result')
plt.plot(np.arange(0,Nsource)/fs_source, s_source)
plt.plot(np.arange(0,N_source_ok)/fs, s_source_ok,  'or', fillstyle='none')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')
#legend('Original source signal', 'Source signal after decimation')
plt.grid()
if PythonConsole:
    print('Close the figure to proceed with deconvolution')
    plt.show(block=True)
    # plt.ion()
    # input('Press ENTER to continue')

if Terminal:
    plt.get_current_fig_manager().window.wm_geometry("600x400+0+800")
    plt.show(block=False)
    input('Press ENTER to proceed with deconvolution')


#--------------------------------------------------------------------------------------
# 6. Deconvolution

NFFT_deconv=max(N_source_ok, N)

sig_f=fft(s, NFFT_deconv)
source_f=fft(s_source_ok, NFFT_deconv)

eee=max(abs(source_f)**2)/100

sig_deconv_f=(sig_f*np.conj(source_f))/np.maximum(abs(source_f)**2,eee) ### deconvolution using Eq (26)
sig_deconv_t=ifft(sig_deconv_f)

# reviewww make real signal ok???not sure

sig_deconv_t_t=np.real(sig_deconv_t)

Nd=NFFT_deconv
time_d=np.arange(0,Nd)/fs

b=np.arange(1,Nd+1)
h=np.hamming(Nw)
rtf_deconv=tfrstft(sig_deconv_t_t, b, NFFT, h)

plt.figure()
print('\n' * 20)
print('The plot shows the spectrogram of the propagated signal after source deconvolution')
print('The signal is now ready for warping. It is saved into css_ready_to_warp.wav')
print('Before warping, let us decimate the signal a bit')

plot=10*np.log10(abs(rtf_deconv)**2)
plt.imshow(plot,extent=[time_d[0], time_d[-1], freq[0], freq[-1]], aspect='auto', origin='low')
plt.ylim([0,fs/2])
plt.xlim([0,0.5])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Received signal after propagation and deconvolution')

if PythonConsole:
    print('Close the figure to proceed with decimation')
    plt.show(block=True)
    # plt.ion()
    # input('Press ENTER to continue')

if Terminal:
    plt.get_current_fig_manager().window.wm_geometry("600x400+800+800")
    plt.show(block=False)
    input('Press ENTER to proceed with decimation')

write('css_ready_to_warp.wav', fs, sig_deconv_t_t)




#--------------------------------------------------------------------------------------
# 7. Decimation

Nsub=10
fs_ok=fs/Nsub
s_ok=signal.decimate(sig_deconv_t,Nsub)
N_ok=len(s_ok)

t_ok=np.arange(0,N_ok)/fs_ok
freq_ok=np.arange(0,NFFT)*fs_ok/NFFT


Nw_ok=51
b=np.arange(1,N_ok+1)
h=np.hamming(Nw_ok)
rtf_ok=tfrstft(s_ok, b, NFFT, h)

print('\n' * 20)
print('Here is the spectrogram of the signal after decimation by a factor of 10')
print('You can now go the the next script to filter the mode using warping')
print('The parameters to use will be')
print('   * decimation rate : 10')
print('   *  sliding window length : 51')
print('  ')

plt.figure()
plot=10*np.log10(abs(rtf_ok)**2)

plt.imshow(plot,extent=[t_ok[0], t_ok[-1], freq_ok[0], freq_ok[-1]], aspect='auto', origin='low')

plt.ylim([0,fs_ok/2])
plt.xlim([0,0.5])

plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Received signal after propagation, deconvolution and decimation')

if PythonConsole:
    print('Close the figure to exit the code')
    plt.show(block=True)
    # plt.ion()
    # input('Press ENTER to continue')

if Terminal:
    plt.get_current_fig_manager().window.wm_geometry("600x400+800+800")
    plt.show(block=False)
    input('Press ENTER to exit the code')



print('  ')
print('END')

