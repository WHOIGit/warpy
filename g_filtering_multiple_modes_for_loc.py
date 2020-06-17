# Warping tutorial
# g_filtering_multiples_modes_for_loc

#May 2020
# Eva Chamorro - Daniel Zitterbart - Julien Bonnel

#--------------------------------------------------------------------------------------
## 1. Import packages

import os
import matplotlib
import scipy.io as sio
from scipy import interpolate
from scipy.signal import hilbert
from subroutines.warping_functions import *
from subroutines.time_frequency_analysis_functions import *
from subroutines.filter import *

'NOTE: This code has to be run in the Terminal'

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
c2 = data['c2']
D = data['D']
rho1 = data['rho1']
rho2 = data['rho2']

#--------------------------------------------------------------------------------------

## 3. Process signal

# The first sample of s_t_dec corresponds to time r/c1

# Make the signal shorter, no need to keep samples with zeros
N_ok=150
s_ok=s_t_dec[:,0:N_ok] ### this is the impulse response of the waveguide

#Corresponding time and frequency axis
time=np.arange(0,N_ok)/fs

# The warped signal will be s_w
[s_w, fs_w]=warp_temp_exa(s_ok,fs,r,c1)
M=len(s_w)

#--------------------------------------------------------------------------------------

## 4. Time frequency representations

# 4.1 Original signal

# STFT computation

NFFT=1024
N_window=31 # you need a short window to see the modes
b=np.arange(1,N_ok+1)
h=np.hamming(N_window)

tfr=tfrstft(s_ok,b,NFFT,h)
spectro=abs(tfr)**2

# Figure
freq=np.arange(0,NFFT)*fs/NFFT

print('This is the spectrogram of the received signal')
print('We will now warp it')


plt.figure()
plt.imshow(spectro, extent=[time[0,0], time[0,-1], freq[0,0], freq[0,-1]],aspect='auto', origin='low')
plt.ylim([0, fs/2])
plt.xlim([0,0.5])  ### Adjust this to see better
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Source signal')
plt.show(block=False)
input('Pres ENTER to continue to warp it')



# 4.2 Warped signal

# STFT computation
N_window_w = 301  # You need a long window to see the warped modes
wind = np.hamming(N_window_w)
wind = wind / np.linalg.norm(wind)
b = np.arange(1, M + 1)
tfr_w = tfrstft(s_w, b, NFFT, wind)
spectro_w = abs(tfr_w) ** 2

# Time and frequency axis of the warped signal
time_w = np.arange(0, (M) / fs_w, 1 / fs_w)
freq_w = np.arange(0, fs_w - fs_w / NFFT + fs_w / NFFT, fs_w / NFFT)

print('\n' * 30)
print('This is the spectrogram of the warped signal')
print('')

# Figure
plt.figure()
plt.imshow(spectro_w, extent=[time_w[0], time_w[-1], freq_w[0], freq_w[-1]], aspect='auto', origin='low')
plt.ylim([0, 40])  ### Adjust this to see better
plt.xlabel('Warped time (sec)')
plt.ylabel('Corresponding warped frequency (Hz)')
plt.title('Warped signal')
plt.show(block=False)
input('Pres ENTER to continue to filtering')


#--------------------------------------------------------------------------------------
## 5. Filtering

# To make it easier, filtering will be done by hand using bbox_select.
# See bbox_selec.py for more information

spectro_w_1=spectro_w[0:200,:]
spectro_w_1=np.transpose(spectro_w_1)
Nmode=4
modes=np.zeros((N_ok,Nmode))
tm=np.zeros((NFFT,Nmode))

print('\n' * 20)
print('Now try to filter the 4 modes')
print('Create the four masks sequentially, starting with mode 1, then 2, etc.')
print('(if needed, go back to c_warping_and_filtering.m for mask creation)')



def pol(arr):

    # create GUI
    app = QtGui.QApplication([])
    w = pg.GraphicsWindow(size=(1000, 800), border=True)
    w.setWindowTitle('pyqtgraph : Filtering')

    text = """Filtering: Now try to filter the 4 modes .<br>\n
    Create the four masks sequentially, starting with mode 1, then 2, etc.<br>\n
    To do so, move the vertices in the image until you are ok with the mask.<br>\n
    You can add a new point by clicking on the blue lines.<br>\n
    Once you are ok with the mask shape, close the image window to continue filtering.<br> \n
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

    # Start Qt event loop unless running in interactive mode or using pyside.
    if __name__ == '__main__':
        import sys

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    mask = roi.getArrayRegion(arr, img1a)
    pts = roi.getState()['points']

    return mask, pts


print('Filtering mode 1')
for i in range (Nmode):

    # Create the mask
    mask, pts = pol(spectro_w_1)
    final_mask = maskc(mask, pts, spectro_w_1)
    mask_ini = np.transpose(final_mask)
    #input('pres enter to continue')

    # add the part masked to the total sprectogram array
    masque_2 = np.zeros_like(spectro_w[200:, :])
    masque = np.concatenate((mask_ini, masque_2), axis=0)

    # Note that the mask is applied on the STFT (not on the spectrogram)
    mode_rtf_warp = masque * tfr_w
    norm = 1 / NFFT / np.max(wind)
    mode_temp_warp = np.real(np.sum(mode_rtf_warp, axis=0)) * norm * 2

    # The modes are actually real quantities, so that they have negative frequencies.
    # Because we do not filter negative frequencies, we need to multiply by 2
    # the positive frequencies we have filter in order to recover the correct
    # modal energy
    mode = iwarp_temp_exa(mode_temp_warp, fs_w, r, c1, fs, N_ok)
    modes[:, i] = mode[:, 0]

    # Verification

    a = hilbert(mode)
    b = np.arange(1, N_ok + 1)
    h = np.hamming(N_window)
    mode_stft = tfrstft(a, b, NFFT, h)
    mode_spectro = abs(mode_stft) ** 2
    tm_1, D2 = momftfr(mode_spectro, 0, N_ok, time)
    tm[:, i] = tm_1[:, 0]
    if i < Nmode:
        print('Filtering mode '+ str(i+2))


print('End of filtering')

#--------------------------------------------------------------------------------------
## 6. Verification

print('\n' * 20)
print('The red lines are the estimated dispersion curves.')
print('Now let us restrict them to a frequency band where they are ok')
print('You will have to enter the min/max frequency for each dispersion curves')
print('(for every mode, choose the widest frequency band over which the dispersion curve estimation looks correct)')


plt.figure()
plt.imshow(spectro, extent=[time[0,0], time[0,-1], freq[0,0], freq[0,-1]],aspect='auto',origin='low')
plt.ylim([0,fs/2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram and estimated dispersion curves')
for i in range(Nmode):
    plt.plot(tm[:,i],freq[0, :],'r')


plt.show(block=False)
input('Pres ENTER to continue to select the frequency band')

#--------------------------------------------------------------------------------------
## 7. Restrict to frequency band of interest

fmin = np.zeros(Nmode)
fmax = np.zeros(Nmode)
for i in range(Nmode):
    fmin[i] = int(input('Enter min freq for mode ' + str(i + 1) + ':  '))
    fmax[i] = int(input('Enter max freq for mode ' + str(i + 1) + ':  '))

tm_ok = tm
freq_sel = freq[0, :]

for i in range(Nmode):
    pos = np.where((freq_sel > fmin[i]) & (freq_sel < fmax[i]))
    poss = np.array(pos, dtype='int')
    tm_ok[0:poss[0, 0], i] = np.nan
    tm_ok[poss[0, -1]:, i] = np.nan

print('\n' * 20)
print('This is the spectrogram with your best guess of the dispersion curves.')


plt.figure()
plt.imshow(spectro, extent=[time[0, 0], time[0, -1], freq[0, 0], freq[0, -1]], aspect='auto', origin='low')
plt.ylim([0, fs / 2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram and estimated dispersion curves')
for i in range(Nmode):
    plt.plot(tm_ok[:, i], freq[0, :], 'r')


plt.show(block=False)
input('Pres ENTER to continue and exit the code')

r_true = r
c1_true = c1
c2_true = c2
rho1_true = rho1
rho2_true = rho2
D_true = D

# let's introduce a small random time shift, as if source and receiver
# were not synchronized

dt_true = np.random.rand(1) * 0.06 - 0.03
tm_for_inv = tm_ok + dt_true

# let's save the tm_for_inv on a relevant frequency axis

fmin_data = min(fmin)
fmax_data = max(fmax)
df_data = 2
freq_data = np.arange(fmin_data, fmax_data + df_data, df_data)
Nf_data = len(freq_data)

data = np.zeros((Nf_data, Nmode))

for i in range(Nmode):
    f = interpolate.interp1d(freq[0, :], tm_for_inv[:, i], bounds_error=False, fill_value=np.nan)
    data[:, i] = f(freq_data)

sio.savemat('data_for_loc.mat', {'r_true': r_true, 'c1_true': c1_true, 'c2_true': c2_true, 'rho1_true': rho1_true,
                         'rho2_true': rho2_true, 'D_true': D_true, 'freq_data': freq_data, 'data': data,
                         'dt_true': dt_true})

print(' ')
print('Your result is saved. If you are happy with it, ')
print('Proceed to the next code for source localization.')
print(' ')
print('END')





