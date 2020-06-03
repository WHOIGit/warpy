## Warping tutorial
#### C_warping_and_filtering

##### April 2020
###### Eva Chamorro - Daniel Zitterbart - Julien Bonnel

#----------------------------------------------------------------------
## 1. Import packages
import os
#os.chdir("/Users/evachamorro/PycharmProjects/WHOI/functions/")
#**Put here the directory where you have the file with your function**
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
#%matplotlib widget
from matplotlib import interactive
from matplotlib import cm
from matplotlib.path import Path
from warping_functions import *
from time_frequency_analysis_functions import *
from bbox_select import *
from scipy.signal import hilbert
from test import *

import warnings
warnings.filterwarnings('ignore')

#os.chdir('/Users/evachamorro/PycharmProjects/WHOI/')
##**Put here the directory where you were working**


#--------------------------------------------------------------------------------------
## 2. Load simulated signal
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

# Select variables
s_t = data['s_t']
fs = data['fs']
s_t_dec = data['s_t_dec']
r = data['r']
c1 = data['c1']


## 3. Process signal

# The first sample of s_t_dec corresponds to time r/c1

# Make the signal shorter, no need to keep samples with zeros
N_ok=150
s_ok=s_t_dec[:,0:N_ok]

# Corresponding time and frequency axis
time=np.arange(0,N_ok)/fs

# The warped signal will be s_w
[s_w, fs_w]=warp_temp_exa(s_ok,fs,r,c1)
M=len(s_w)


## 4. Time frequency representations

### 4.1 Original signal

### Original signal
# STFT computation
NFFT=1024
N_window=31 ### you need a short window to see the modes
b=np.arange(1,N_ok+1)
b=b[np.newaxis,:]
d=np.hamming(N_window)
d=d[:,np.newaxis]
tfr=tfrstft(s_ok,b,NFFT,d)
spectro=abs(tfr)**2


# Figure

print('This is the spectrogram of the received signal')
print('We will now warp it')

freq=(np.arange(0,NFFT))*fs/NFFT

plt.figure()
plt.imshow(spectro, extent=[time[0,0], time[0,-1], freq[0,0], freq[0,-1]],aspect='auto',origin='low')
plt.ylim([0, fs/2])
plt.xlim([0, 0.5])  ### Adjust this to see better
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Original signal')
plt.show(block=True)

input('Pres ENTER to Continue and proceed with warping')

### 4.2 Warped signal

## Warped signal

# STFT computation

N_window_w=301 # You need a long window to see the warped modes
wind=np.hamming(N_window_w)
wind=wind/np.linalg.norm(wind)
wind=wind[:,np.newaxis]
t=np.arange(1,M+1)
t=t[np.newaxis,:]

tfr_w=tfrstft(s_w,t,NFFT,wind)

spectro_w=abs(tfr_w)**2

## Time and Frequency axis of the warped signal
time_w=np.arange(0,(M)/fs_w,1/fs_w)
freq_w=np.arange(0,NFFT)*fs/NFFT


## selection the spectro_w part to mask
spectro_w_1=spectro_w[0:200,:]
spectro_w_1=np.transpose(spectro_w_1)

# Figure

print('This is the spectrogram of the warped signal')
print('')

plt.figure()
plt.imshow(spectro_w, extent=[time_w[0], time_w[-1], freq_w[0,0], freq_w[0,-1]],aspect='auto',origin='low' )
plt.ylim([0,40])
plt.xlabel('Warped time (sec)')
plt.ylabel('Corresponding warped frequency (Hz)')
plt.title('Warped signal')
plt.show(block=True)

print('Continue to filtering')


## 5. Filtering

# To make it easier, filtering will be done by hand using the pyqtgraph package.
# See pyqtgraph ROI help for more information

print('We will now filter a single mode by creating a time-frequency mask.')
print('To do so, move the vertices in the image until you are ok with the mask. You can add a new vertice with a click on the blue lines')
print('Once you are ok with the mask shape, close the image window')
print('Look at Fig. 11 in the paper for a mask shape suggestion')

from maskc_t import *
####### func ########3
def pol(arr):

    ## create GUI
    app = QtGui.QApplication([])
    w = pg.GraphicsWindow(size=(1000, 800), border=True)
    w.setWindowTitle('pyqtgraph example: ROI Examples')

    text = """Data Selection From Image.<br>\n
    Draw a mask to filter one mode.<br>\n
    To do so, move the vertices in the image until you are ok with the mask.<br>\n
    You can add a new point by clicking on the blue lines.<br>\n
    Once you are ok with the mask shape, close the image window.<br> \n
    Be careful not to get out of the image """

    w1 = w.addLayout(row=0, col=0)
    label1 = w1.addLabel(text, row=0, col=0)
    v1a = w1.addViewBox(row=1, col=0, lockAspect=True)
    img1a = pg.ImageItem(arr)

    # Get the colormap
    colormap = cm.get_cmap("viridis")  # cm.get_cmap("CMRmap")
    colormap._init()
    lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

    # Apply the colormap
    img1a.setLookupTable(lut)

    v1a.addItem(img1a)

    v1a.disableAutoRange('xy')

    v1a.autoRange()

    rois = []

    rois.append(pg.PolyLineROI([[80, 60], [90, 30], [60, 40]], pen=(6, 9), closed=True))

    def update(roi):
        roi.getArrayRegion(arr, img1a)

    for roi in rois:
        roi.sigRegionChanged.connect(update)
        v1a.addItem(roi)

    ## Start Qt event loop unless running in interactive mode or using pyside.
    if __name__ == '__main__':
        import sys

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    mask = roi.getArrayRegion(arr, img1a)
    pts = roi.getState()['points']

    return mask, pts


maska, pts = pol(spectro_w_1)
final_mask = maskc(maska, pts,spectro_w_1 )
mask_ini=np.transpose(final_mask)

mask_ini=np.transpose(final_mask)

# add the part masked to the total sprectogram array
masque_2=np.zeros_like(spectro_w[200:,:])
masque=np.concatenate((mask_ini,masque_2),axis=0)

# Note that the mask is applied on the STFT (not on the spectrogram)
mode_rtf_warp=masque*tfr_w
norm=1/NFFT/np.max(wind)
mode_temp_warp=np.real(np.sum(mode_rtf_warp,axis=0))*norm*2

# The modes are actually real quantities, so that they have negative frequencies.
# Because we do not filter negative frequencies, we need to multiply by 2
# the positive frequencies we have filter in order to recover the correct
# modal energy
mode=iwarp_temp_exa(mode_temp_warp,fs_w,r,c1,fs,N_ok)

input('Pres ENTER to Continue and proceed with verification')

## 6. Verification

# you can estimate the dispersion curve by computing the
# frequency moment of the filtered mode TFR

a=hilbert(mode)
b=np.arange(1,N_ok+1)
b=b[np.newaxis,:]
d=np.hamming(N_window)
d=d[:,np.newaxis]

mode_stft=tfrstft(a,b,NFFT,d)
mode_spectro=abs(mode_stft)**2
tm,D2=momftfr(mode_spectro,0,N_ok,time)


print('The red line is the estimated dispersion curve.')
print('You have to restrict it to a frequency band where it is relevant: the red line')
print('is supposed to follow the mode for frequencies where the mode has a high amplitude,')
print('but it can look noisy at frequencies where the mode has a low amplitude (e.g. at frequencies below cutoff) ')
print('...')
print('If the result is not satisfying, you must create a new filtering mask')
print('If the result is ok, you can try another mode!')

plt.figure()
plt.imshow(spectro, extent=[time[0,0], time[0,-1], freq[0,0], freq[0,-1]],aspect='auto',origin='low')
plt.ylim([0,fs/2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram and estimated dispersion curve')
plt.plot(tm[:,0],freq[0, :],'r')
plt.show(block=True)

print('Continue to look at your result vs the true modes')
input('Pres ENTER to Continue to look at your result vs the true modes')

## 7. Last verification

data = sio.loadmat(os.getcwd()+ '/sig_pek_and_modes_for_warp.mat')
r=data['r']
vg=data['vg']
c1=data['c1']
f_vg=data['f_vg']


### creation of the theoretical dispersion curves
tm_theo=r/vg-r/c1 ### range over group_speed minus correction for time origin


print('This is the same figure than before, except that the true dispersion curves are now in black.')
print('How well did you do?')
print(' ')
print('Recall that you have to restrict interpretation of the dispersion curve (red line)')
print('to only the frequency band where it is relevant. The black and red lines will not match')
print('entirely for the length of the red line')
print(' ')

plt.figure()
plt.imshow(spectro, extent=[time[0,0], time[0,-1], freq[0,0], freq[0,-1]],aspect='auto',origin='low')
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
plt.show(block=True)

print(' ')
print('END')