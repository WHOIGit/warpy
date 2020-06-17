# Warping tutorial
# D_playing_with_time_origin

# May 2020
# Eva Chamorro - Daniel Zitterbart - Julien Bonnel

#--------------------------------------------------------------------------------------
## 1. Import packages

import os
import matplotlib
from matplotlib import interactive

import scipy.io as sio
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

# IF YOU CHANGE RANGE, you must change the following variable which
# defines the bounds on the x-axis for all the plots

xlim_plots = [6.5, 7.5]



#--------------------------------------------------------------------------------------
## 3. Plot spectrogram and time origin selection

N = len(s_t[0, :])
NFFT = 2048  # FFT size
N_window = 31  # sliding window size (need an odd number)

redo_dt = 0

while redo_dt == 0:

    a = np.arange(1, N + 1)
    d = np.hamming(N_window)
    tfr = tfrstft(s_t, a, NFFT, d)
    spectro = abs(tfr) ** 2

    time = np.arange(0, N) / fs
    freq = np.arange(0, NFFT) * fs / NFFT


    print('This is the spectrogram of a signal propagated in a Pekeris waveguide')
    print('In this code, you will choose the time origin for warping')
    print('Double-click once on the spectrogram at the position where you want to define the time origin')


    plt.figure()
    plt.imshow(spectro, extent=[time[0, 0], time[0, -1], freq[0, 0], freq[0, -1]], aspect='auto')
    plt.ylim([0, fs / 2])
    plt.xlim(xlim_plots)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.show(block=False)

    interactive(True)
    t_dec = plt.ginput(1)
    print('\n' * 30)
    input('Pres ENTER to continue and plot the result')
    t_dec = t_dec[0]
    t_dec = t_dec[0]

    # 4. Shorten the signal, make it start at the chosen time origin, and warp it
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

    # 5. Plot spectrograms

    # Original signal
    N_window = 31  # you need a short window to see the modes

    a = np.arange(1, N_ok + 1)
    d = np.hamming(N_window)
    tfr = tfrstft(s_ok, a, NFFT, d)
    spectro = abs(tfr) ** 2

    # Warped signal
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
    print('The left panel shows the spectrogram of the original ')
    print('signal with the chosen time origin')
    print('The right panel shows the corresponding warped signal')


    # Figure
    plt.figure(figsize=(8, 6))
    plt.subplot(121)
    plt.imshow(spectro, extent=[time_ok[0, 0], time_ok[0, -1], freq[0, 0], freq[0, -1]], aspect='auto')
    plt.ylim([0, fs / 2])
    # plt.xlim([0, 0.5])  ### Adjust this to see better
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Original signal with chosen time origin')

    # Figure
    plt.subplot(122)
    plt.imshow(spectro_w, extent=[time_w[0], time_w[-1], freq_w[0], freq_w[-1]], aspect='auto')
    plt.ylim([0, 40])  ### Adjust this to see better
    plt.xlabel('Warped time (sec)')
    plt.ylabel('Corresponding warped frequency (Hz)')
    plt.title('Corresponding warped signal')
    plt.show(block=False)
    input('Pres ENTER to continue')

    print('\n' * 30)
    print('Type 0 if you want to redo the time origin selection')
    redo_dt = input('or type 1 if you want to proceed with modal filtering ')

    if redo_dt == '' or redo_dt == '1':
        redo_dt = 1

    elif redo_dt == '0':
        redo_dt = int(redo_dt)




#--------------------------------------------------------------------------------------
## 6. Filtering

# selection the spectro_w part to mask
spectro_w_1=spectro_w[0:400,:]
spectro_w_1=np.transpose(spectro_w_1)
time_w_1=time_w
freq_w_1=freq_w[:400]



# To make it easier, filtering will be done by hand using the pyqtgraph ROI tool.
# See pyqtgraph help for more information

print('\n' * 30)
print('Now try to filter the 4 modes')
print('Create the four masks sequentially, starting with mode 1, then 2, etc.')
print('To do so, move the vertices in the image until you are ok with the mask. You can add a new vertice with a click on the blue lines')
print('Once you are ok with the mask shape of one mode, close the image window to continue filtering')
print('(if needed, go back to c_warping_and_filtering.m for mask creation)')
print('Look at Fig. 11 in the paper for a mask shape suggestion')
print('(you can enlarge the figure or make it full screen before creating the mask)')

Nmodes=4
modes=np.zeros((N_ok,Nmodes))
tm=np.zeros((NFFT,Nmodes))



def pol(arr):

    # create GUI
    app = QtGui.QApplication([])
    w = pg.GraphicsWindow(size=(1000, 800), border=True)
    w.setWindowTitle('pyqtgraph: Filtering')

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
for i in range (Nmodes):

    # Create the mask
    mask, pts = pol(spectro_w_1)
    final_mask = maskc(mask, pts, spectro_w_1)
    mask_ini = np.transpose(final_mask)

    # add the part masked to the total sprectogram array
    masque_2 = np.zeros_like(spectro_w[400:, :])
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
    tm_1, D2 = momftfr(mode_spectro, 0, N_ok, time_ok)
    tm[:, i] = tm_1[:, 0]
    if i < Nmodes:
        print('Filtering mode '+ str(i+2))



print('End of filtering')



#--------------------------------------------------------------------------------------
## 7. Verification
print('\n' * 30)
print('The red lines are the estimated dispersion curves.')
print(' ')


plt.figure(figsize=[7,5])
plt.imshow(spectro, extent=[time_ok[0,0], time_ok[0,-1], freq[0,0], freq[0,-1]],aspect='auto',origin='low')
plt.ylim([0,fs/2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram and estimated dispersion curve')
for i in range (Nmodes):
    plt.plot(tm[:,i],freq[0, :],'r')

plt.show(block=False)
input('Pres ENTER to continue to look at your result vs the true modes')





#--------------------------------------------------------------------------------------
## 8. Last verification

data = sio.loadmat(os.getcwd()+ '/sig_pek_and_modes_for_warp.mat')
r=data['r']
vg=data['vg']
c1=data['c1']
f_vg=data['f_vg']


# creation of the theoretical dispersion curves
tm_theo=r/vg-t_dec ### range over group_speed minus correction for time origin
print('\n' * 30)
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



plt.figure(figsize=[7,5])
plt.imshow(spectro, extent=[time_ok[0,0], time_ok[0,-1], freq[0,0], freq[0,-1]],aspect='auto',origin='low')
plt.ylim([0,fs/2])
plt.xlim([0,0.74])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram and estimated dispersion curve')
for i in range (Nmodes):
    plt.plot(tm_theo[i,:], f_vg[0,:], 'black')
    plt.plot(tm[:,i],freq[0, :],'r')

plt.show(block=False)
input('Pres ENTER to exit the code')


print(' ')
print('END')
