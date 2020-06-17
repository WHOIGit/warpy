# Warping tutorial
# a_look_at_data, upsweep

# April 2020
# Eva Chamorro - Daniel Zitterbart - Julien Bonnel

#--------------------------------------------------------------------------------------
## 1. Import packages

import os
import sys
import wave
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as siw
import scipy.signal as sig
sys.path.insert(0, os.path.dirname(os.getcwd())+'/subroutines') #**Put here the directory where you have the file with your function**
from warping_functions import *
from time_frequency_analysis_functions import *

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

PythonConsole=0
Terminal=1

if Terminal:
    matplotlib.use("TkAgg")

if PythonConsole == Terminal:
    raise ValueError ('Both environment cannot be activated/deactivated at the same time')



#--------------------------------------------------------------------------------------
## 2. Load and decimate

# Option to plot spectrogram in dB vs linear scale
plot_in_dB = 1  ### 1 to plot in dB, or 0 to plot in linear scale

# Parameters for computing/plotting the spectrograms
# These should be good default values for the gunshot provided in this tutorial
# You will likely have to change them for other data
NFFT = 2048  ### fft size for all spectrogram computation

print('\n' * 20)
print('Select a wav file with the signal you want to analyze')
print(
    'Ideally, you have created this wav file yourself using any audio/bioacoustics software, and you know its content')
print('  * the wav file should be a small data snippet (say, a few seconds) that contains the relevant signal')
print('  * the was file must contain a single channel')
print('  * you must know the maximum frequency of interest for the signal under study (f_max)')
print('    (you can assess this maximum frequency though a preliminary spectrogram analysis on another audio software)')
print('Now select the wav file')

path = os.getcwd() + '/upsweep.wav'
fs0, file = siw.read(path)
f = wave.open(os.path.join(path, path), 'rb')

if f.getnchannels() != 1:
    raise ValueError('The input wav file must contain a single channel')

y0 = file
y0 = y0 - np.mean(y0)
N0 = len(y0)

print(['The sampling frequency of your signal is fs=' + str(fs0) + ' Hz'])
print('If fs > 4*fmax, you should decimate your signal, i.e. reducing the sampling by a factor of N')
print('(after decimation, the new sampling frequency is thus fs/N)')
print(
    'If you decimate, you must choose N so that the new sampling frequency is as small as possible, but nonetheless greater than 2*fmax')
print(' ')
print('Do you want to decimate your signal?')
print('If you do not want to decimate your signal type 1')
print('If you want to decimate your signal, enter the decimation rate')
subsamp = input('(for the upsweep provided in the tutorial, enter 18); ');

if len(subsamp) == 0:
    subsamp = 1

else:
    subsamp = int(subsamp)

s = sig.decimate(y0, subsamp)
Ns = len(s)
fs = fs0 / subsamp

del fs0, y0

#--------------------------------------------------------------------------------------
## 3. Figure

freq = np.arange(0, NFFT + 2) * fs / NFFT
time = np.arange(0, Ns) / fs

N_window = 71
b = np.arange(1, Ns + 1)
h = np.hamming(N_window)
tfr = tfrstft(s, b, NFFT, h)

if plot_in_dB == 1:
    spectro = 20 * np.log10((abs(tfr)))
    plt.imshow(spectro, extent=[time[0], time[-1], freq[0], freq[-1]], aspect='auto', origin='low')

else:
    spectro = np.abs(tfr) ** 2
    plt.imshow(spectro, extent=[time[0], time[-1], freq[0], freq[-1]], aspect='auto', origin='low')

print('\n' * 20)
print('This is the spectrogram, computed with a sliding window of length 71 samples')

plt.ylim([0, fs / 2])
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('Spectrogram')

if PythonConsole:
    print('Close the figure to continue and modify the window length')
    plt.show(block=True)
    # plt.ion()
    # input('Press ENTER to continue')

if Terminal:
    plt.get_current_fig_manager().window.wm_geometry("600x400+0+0")
    plt.show(block=False)
    input('Press ENTER to continue and modify the window length')


#--------------------------------------------------------------------------------------
## 4. Modify the window length


print('Now you can modify the window length')
print('Remember to give it in number of samples')
print(' ')


while ((N_window != 0)):
    print('Input window length and press enter for spectrogram computation : ')
    N_window = int(input('(window length 0 to exit the code) '))
    print('\n' * 20)
    if N_window == 0:
        break

    if  N_window != 0:
        # STFT computation
        h = np.hamming(N_window)
        tfr = tfrstft(s, b, NFFT, h)

        # Spectrogram ~ modulus STFT
        spectro = abs(tfr) ** 2

        plt.figure()
        if plot_in_dB == 1:
            plt.imshow(10 * np.log10(spectro), extent=[time[0], time[-1], freq[0], freq[-1]], aspect='auto', origin='low')
        else:
            s = spectro
            plt.imshow(s, extent=[time[0], time[-1], freq[0], freq[-1]], aspect='auto', origin='low')

        plt.ylim([0, fs / 2])
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram N_window: ' + str(N_window))
        if PythonConsole:
            print('Close the figure to continue to modify the window length or exit the code')
            plt.show(block=True)
            # plt.ion()
            # input('Press ENTER to continue')

        if Terminal:
            plt.get_current_fig_manager().window.wm_geometry("600x400+800+0")
            plt.show(block=False)
            input('Pres ENTER to continue to modify the window length or exit the code')

print(['As a reminder, you chose a decimation rate N=' + str(subsamp)])
print('Remember this number for the next script')

print('  ')
print('END')
