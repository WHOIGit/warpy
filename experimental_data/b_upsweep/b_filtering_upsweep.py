# Warping tutorial
# b_filtering, upsweep

# May 2020
# Eva Chamorro - Daniel Zitterbart - Julien Bonnel

#--------------------------------------------------------------------------------------
## 1. Import packages

import os
import sys
import wave
from matplotlib import interactive
import scipy.io as sio
from scipy.fftpack import ifft
import scipy.io.wavfile as siw
import scipy.signal as sig
from scipy.signal import hilbert
from scipy import interpolate
from datetime import date
sys.path.insert(0, os.path.dirname(os.getcwd())+'/subroutines') #**Put here the directory where you have the file with your function**
from warping_functions import *
from time_frequency_analysis_functions import *
from filter import *

import warnings
warnings.filterwarnings('ignore')

'NOTE: This code has to be run in the Terminal'

#--------------------------------------------------------------------------------------
## 2. Load and decimate

# Option to plot spectrogram in dB vs linear scale
plot_in_dB = 0  ### 1 to plot in dB, or 0 to plot in linear scale

# Parameters for computing/plotting the spectrograms
# These should be good default values for the gunshot provided in this tutorial
# You will likely have to change them for other data
NFFT = 2048  ### fft size for all spectrogram computation
N_window_w = 301  ###length of the sliding window for the spectrogram of the warped signal
fmax_plot = 90  ### max frequency for the plot of the spectrogram of the warped signal

# Parameters for warping
# You can keep these default value or change them if you have reasons to know better
r = 10000
c1 = 1500

# Parameters for source deconvolution
n_click = 4  ##number of clicks to do the source deconvolution


print('\n' * 20)
print('Select a wav file with the signal you want to analyze')
print('This wav file should be a small data snippet (say, a few seconds) that contains the relevant signal')
print('The was file must contain a single channel')
print('You should have run the code a_look_at_data.m only once and you should know')
print('adequate value of decimation rate and window length to compute spectrograms')
print('Now select the wav file')

path = os.getcwd() + '/upsweep.wav'
fs0, file = siw.read(path)
f = wave.open(os.path.join(path, path), 'rb')

if f.getnchannels() != 1:
    raise ValueError('The input wav file must contain a single channel')

y0 = file
y0 = y0 - np.mean(y0)
N0 = len(y0)

print('')
print('Do you want to decimate your signal?')
print('If you do not want to decimate your signal type 1')
print('If you want to decimate your signal, enter the decimation rate')
subsamp = input('(for the upsweep provided in the tutorial, enter 18); ')

if len(subsamp) == 0:
    subsamp = 18

else:
    subsamp = int(subsamp)

s_t = sig.decimate(y0, subsamp)
Ns = len(s_t)
fs = fs0 / subsamp

del fs0, y0

print('What is an appropriate window length to compute the spectrogram?')
N_window = input('(for the upsweep provided in the tutorial, enter 61); ')

if len(N_window) == 0:
    N_window = 61
else:
    N_window = int(N_window)

if (N_window % 2) == 0:
    N_window = N_window + 1  ## force odd numbers

#--------------------------------------------------------------------------------------
## 3. Plot spectrogram and time origin selection

N = len(s_t)
redo_dt = 0

while redo_dt == 0:

    print('\n' * 20)
    print('This is the spectrogram of your signal')
    print('Click twice on the spectrogram to select the interesting part:')
    print('   * the first defines the time origin for warping')
    print('   * the second click defines the end of the signal that will be warped')
    print('(keep a few hundred ms of noise after the end of the signal of interest)')
    print('The min/max frequencies will be used only to restric the spectrogram plots')
    print('(keep a 5-10 Hz margin around your signal of interest)')
    print('')

    b = np.arange(1, N + 1)
    h = np.hamming(N_window)

    tfr = tfrstft(s_t, b, NFFT, h)
    spectro = np.abs(tfr) ** 2

    time = np.arange(0, N) / fs
    freq = np.arange(0, NFFT) * fs / NFFT

    plt.figure()
    if plot_in_dB == 1:
        plt.imshow(10 * np.log10(spectro), extent=[time[0], time[-1], freq[0], freq[-1]], aspect='auto', origin='low')

    else:
        plt.imshow(spectro, extent=[time[0], time[-1], freq[0], freq[-1]], aspect='auto', origin='low')

    plt.ylim([0, fs / 2])
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram')
    plt.show(block=False)

    interactive(True)
    pts = plt.ginput(2)
    input('Pres ENTER to continue and plot the result')
    c = np.array(pts)
    ttt = c[:, 0]
    fff = c[:, 1]


    # we extract the selected time and frequency min and máx.
    time_origin = ttt[0]
    time_t_dec = np.abs(time - time_origin)
    ind0 = np.where(time_t_dec == np.min(time_t_dec))
    ind0 = ind0[0]
    ind0 = ind0[0]

    freq_origin = fff[0]
    freq_f_dec = np.abs(freq - freq_origin)
    indf0 = np.where(freq_f_dec == np.min(freq_f_dec))
    indf0 = indf0[0]
    indf0 = indf0[0]

    time_end = ttt[1]
    time_t_dec = np.abs(time - time_end)
    ind1 = np.where(time_t_dec == np.min(time_t_dec))
    ind1 = ind1[0]
    ind1 = ind1[0]

    freq_end = fff[1]
    freq_f_dec = np.abs(freq - freq_end)
    indf1 = np.where(freq_f_dec == np.min(freq_f_dec))
    indf1 = indf1[0]
    indf1 = indf1[0]

    t_dec = [time_origin, time_end]
    f_dec = [freq_origin, freq_end]

    s_ok = s_t[ind0:ind1]
    N_ok = len(s_ok)

    # Corresponding time and frequency axis
    time_ok = np.arange(0, N_ok) / fs

    # The warped signal will be s_w
    fs_1 = np.array([[fs]])
    r_1 = np.array([[r]])
    c1_1 = np.array([[c1]])
    [s_w, fs_w] = warp_temp_exa(s_ok, fs_1, r_1, c1_1)

    M = len(s_w)

    # 4. Plot spectrograms

    # Original signal

    b = np.arange(1, N_ok + 1)
    h = np.hamming(N_window)
    tfr = tfrstft(s_ok, b, NFFT, h)
    spectro = abs(tfr) ** 2

    # Warped signal
    wind = np.hamming(N_window_w)
    wind = wind / np.linalg.norm(wind)
    b = np.arange(1, M + 1)
    tfr_w = tfrstft(s_w, b, NFFT, wind)
    spectro_w = abs(tfr_w) ** 2

    # Time and frequency axis of the warped signal
    time_w = np.arange(0, M / fs_w, 1 / fs_w)
    freq_w = np.arange(0, fs_w - fs_w / NFFT + fs_w / NFFT, fs_w / NFFT)

    # Figure
    print('\n' * 20)
    print('This is the signal you selected')

    plt.figure()

    if plot_in_dB == 1:
        plt.imshow(10 * np.log10(spectro), extent=[time[0], time[-1], freq[0], freq[-1]], aspect='auto', origin='low')

    else:
        plt.imshow(spectro, extent=[time[0], time[-1], freq[0], freq[-1]], aspect='auto', origin='low')

    plt.ylim([f_dec[0], f_dec[1]])
    # plt.xlim([0, 0.5])  ### Adjust this to see better
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Original signal with chosen time origin')
    plt.show(block=False)

    print('Type 0 if you want to redo the time origin selection')
    redo_dt = input('or type 1 if you want to proceed with modal filtering ')

    if redo_dt == '' or redo_dt == '1':
        redo_dt = 1

    elif redo_dt == '0':
        redo_dt = int(redo_dt)




spectro0 = spectro
time0 = time_ok
N0 = N_ok

#--------------------------------------------------------------------------------------
## 4.Phase compensation

print('\n' * 20)
print('This is the same spectrogram than earlier, we will now proceed with phase compensation')
print(['Once again, we will approximate the source TF law with ' + str(n_click-1) +' linear pieces.'])
print('As explained in Sec V C, an easy way to do so is to roughly follow the TF contour of mode 1')
print(['Now, click  ' +str(n_click) +'  times on the spectrogram to define your ' +str(n_click-1)+' linear pieces'])
print('(if you work on the upsweep provided in the tutorial, the easiest is probably to start at early-times/low')
print('frequencies, and to progress toward increasing times and increasing frequencies')
print('(roughly follow the red line from Fig 16)')

flag_pc=0

while flag_pc == 0:

    plt.figure()
    print('\n' * 20)
    print('Click 4 times on the spectrogram to define your 3 linear pieces')
    print('In this case, the easiest is probably to start at early-times/high frequencies,')
    print('and to progress toward increasing times and decreasing frequencies')
    print('(roughly follow the black line from the previous example)')

    if plot_in_dB == 1:
        plt.imshow(10 * np.log10(spectro), extent=[time_ok[0], time_ok[-1], freq[0], freq[-1]], aspect='auto',
                   origin='low')
    else:
        plt.imshow(spectro, extent=[time_ok[0], time_ok[-1], freq[0], freq[-1]], aspect='auto',
                   origin='low')

    plt.ylim([f_dec[0], f_dec[1]])
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Received signal')
    plt.show(block=False)

    interactive(True)
    pts = plt.ginput(4)
    input('Pres ENTER to continue and plot the result')

    c = np.array(pts)
    ttt = c[:, 0]
    fff = c[:, 1]

    #% % % Sort
    #elements in chronological
    #time
    #[ttt, ind_sort] = sort(ttt);
    #fff = fff(ind_sort);

    # convert to samples and reduced frequencies
    ttt_s = np.round(ttt * fs_1) + 1
    fff_s = fff / fs_1
    n_click = 4

    # create the piecewise linear-FM source
    for ii in range(1, n_click):
        ifl = np.linspace(fff_s[0, ii - 1], fff_s[0, ii], int(ttt_s[0, ii] - ttt_s[0, ii - 1]))

        if ii == 1:
            iflaw = ifl
        else:
            iflaw = np.concatenate((iflaw, ifl[1:]))

    iflaw = iflaw[:, np.newaxis]

    source_est, IFLAW_est = fmodany(iflaw)
    source_est = source_est[:, np.newaxis]
    source_est_f = fft(source_est, N_ok, axis=0)  ### estimated source signal in the frequency domain
    phi = np.angle(source_est_f)  ### phase of the estimated source signal in the frequency domain

    print('\n' * 20)
    print('The black line is your best guess of the source signal')
    print('Now, let us do phase compensation to transform our received signal')
    print('into something that looks like the impulse response of the waveguide')

    # let's plot it on top of the received signal
    plt.figure()
    if plot_in_dB == 1:
        plt.imshow(10 * np.log10(spectro), extent=[time_ok[0], time_ok[-1], freq[0], freq[-1]], aspect='auto',
                   origin='low')

    else:
        plt.imshow(spectro, extent=[time_ok[0], time_ok[-1], freq[0], freq[-1]], aspect='auto', origin='low')

    plt.ylim([0, fs / 2])
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Received signal')
    plt.plot(ttt, fff, 'black', linewidth=3)
    plt.show(block=False)
    input('Press ENTER to continue')

    # Phase correction
    s_ok_1=s_ok[np.newaxis,:]
    sig_prop_f = fft(s_ok_1, axis=1)
    i = complex(0, 1)
    sig_rec_f = sig_prop_f * np.transpose(np.exp(-i * phi))  # note that only source phase is deconvoluted
    sig_rec_t = ifft(sig_rec_f, axis=1)  # Signal in time domain after source deconvolution

    # Figure
    b = np.arange(1, N_ok + 1)
    d = np.hamming(N_window)

    tfr_sig_rec = tfrstft(sig_rec_t, b, NFFT, d)

    print('Here is the result')
    plt.figure()
    if plot_in_dB == 1:
        plt.imshow(10*np.log10(abs(tfr_sig_rec) ** 2), extent=[time_ok[0], time_ok[-1], freq[0], freq[-1]], aspect='auto',
                   origin='low')
    else:
        plt.imshow(abs(tfr_sig_rec) ** 2, extent=[time_ok[0], time_ok[-1], freq[0], freq[-1]], aspect='auto',
                   origin='low')


    plt.ylim([0, fs / 2])
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Signal after phase compensation')
    plt.show(block=False)
    print('\n' * 20)
    print('If you want to redo the phase compensation, press 0')
    print('The upsweep provided in the tutorial is a tricky signal, you will likely have to try multiple time (downsweeps are easier)')
    flag_pc = input('(you will have the opportunity to modify the time origin later) ')

    if flag_pc == '' or flag_pc == '1':
        flag_pc = 1

    elif flag_pc == '0':
        flag_pc = int(flag_pc)


#--------------------------------------------------------------------------------------
## 5. Play with time origin

redo_dt = 0

while redo_dt == 0:

    print('\n' * 20)
    print('You must now define the time origin')
    print('Click once on the spectrogram at the position where you want to define the time origin')

    plt.figure()
    if plot_in_dB == 1:
        plt.imshow(10 * np.log10(abs(tfr_sig_rec)**2), extent=[time[0], time[-1], freq[0], freq[-1]], aspect='auto', origin='low')

    else:
        plt.imshow(abs(tfr_sig_rec)**2, extent=[time[0], time[-1], freq[0], freq[-1]], aspect='auto', origin='low')


    plt.ylim([f_dec[0],f_dec[1]])
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Signal after phase compensation')
    plt.show(block=False)

    interactive(True)
    pts = plt.ginput(1)
    input('Pres ENTER to continue and plot the result')
    c = np.array(pts)
    ttt = c[:, 0]


    time_origin = ttt[0]
    time_t_dec = np.abs(time - time_origin)
    ind0 = np.where(time_t_dec == np.min(time_t_dec))
    ind0 = ind0[0]
    ind0 = ind0[0]


    t_dec2 = time_origin

    s_ok = s_t[ind0:ind1]
    N_ok = len(s_ok)

    # Corresponding time and frequency axis
    time_ok = np.arange(0, N_ok) / fs

    # The warped signal will be s_w
    fs_1 = np.array([[fs]])
    r_1 = np.array([[r]])
    c1_1 = np.array([[c1]])
    [s_w, fs_w] = warp_temp_exa(s_ok, fs_1, r_1, c1_1)

    M = len(s_w)

    #  Plot spectrograms

    # Original signal
    b = np.arange(1, N_ok + 1)
    h = np.hamming(N_window)
    tfr = tfrstft(s_ok, b, NFFT, h)
    spectro = abs(tfr) ** 2

    # Warped signal
    wind = np.hamming(N_window_w)
    wind = wind / np.linalg.norm(wind)
    b = np.arange(1, M + 1)
    tfr_w = tfrstft(s_w, b, NFFT, wind)
    spectro_w = abs(tfr_w) ** 2

    # Time and frequency axis of the warped signal
    time_w = np.arange(0, M / fs_w, 1 / fs_w)
    freq_w = np.arange(0, fs_w - fs_w / NFFT + fs_w / NFFT, fs_w / NFFT)

    # Figure
    print('\n' * 20)
    print('The left panel shows the spectrogram of the original signal with the chosen time origin')
    print('The right panel shows the corresponding warped signal')
    print('')

    plt.figure()
    plt.subplot(121)

    if plot_in_dB == 1:
        plt.imshow(10 * np.log10(spectro), extent=[time[0], time[-1], freq[0], freq[-1]], aspect='auto', origin='low')

    else:
        plt.imshow(spectro, extent=[time[0], time[-1], freq[0], freq[-1]], aspect='auto', origin='low')

    plt.ylim([f_dec[0],f_dec[1]])
    # plt.xlim([0, 0.5])  ### Adjust this to see better
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Original signal with chosen time origin')

    plt.subplot(122)

    if plot_in_dB == 1:
        plt.imshow(10 * np.log10(spectro_w), extent=[time_w[0], time_w[-1], freq_w[0], freq_w[-1]], aspect='auto', origin='low')

    else:
        plt.imshow(spectro_w, extent=[time_w[0], time_w[-1], freq_w[0], freq_w[-1]], aspect='auto', origin='low')

    plt.ylim([0, fmax_plot])  ### Adjust this to see better
    plt.xlabel('Warped time (sec)')
    plt.ylabel('Corresponding warped frequency (Hz)')
    plt.title('Corresponding warped signal')
    plt.show(block=False)

    print('\n' * 20)
    print('Type 0 if you want to redo the time origin selection')
    redo_dt = input('or type 1 if you want to proceed with modal filtering ')

    if redo_dt == '' or redo_dt == '1':
        redo_dt = 1

    elif redo_dt == '0':
        redo_dt = int(redo_dt)


#--------------------------------------------------------------------------------------
## 6.  Filtering

print('\n' * 20)
print('How many mode do you want to filter? ')
Nmodes=input('(for the upsweep provided in the tutorial, enter 3) ');
if len(Nmodes)==0:
    Nmodes=3
else:
    Nmodes=int(Nmodes)


print('Remember that you can modify the size of the picture before creating the mask')
print('(it is usually a good idea to make the figure full screen to facilitate masking)')


# selection the spectro_w part to mask
spectro_w_1=spectro_w[0:400,:]
spectro_w_1=np.transpose(spectro_w_1)
time_w_1=time_w
freq_w_1 = freq_w[:400]

modes = np.zeros((N_ok, Nmodes))
tm = np.zeros((NFFT, Nmodes))
mode_pts=[]

# Filtering will be done by hand using the pyqtgraph ROI tool.
# See pyqtgraph help for more information


def pol(arr):

    # create GUI
    app = QtGui.QApplication([])
    w = pg.GraphicsWindow(size=(1000, 500), border=True)
    w.setWindowTitle('pyqtgraph: Filtering')

    text = """Filtering: Now try to filter the modes .<br>\n
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

    rois.append(pg.PolyLineROI([[1400, 200], [1600, 200], [1600, 300], [1400, 300]], pen=(6, 9), closed=True))

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


for i in range (Nmodes):

    # Create the mask
    if plot_in_dB==1:
        arr=10 * np.log10(spectro_w_1)
        mask,pts = pol(arr)
    else:
        mask, pts = pol(spectro_w_1)


    final_mask = maskc(mask, pts, spectro_w_1)
    mask_ini = np.transpose(final_mask)

    # add the part masked to the total sprectogram array
    masque_2 = np.zeros_like(spectro_w[400:, :])
    masque = np.concatenate((mask_ini, masque_2), axis=0)

    #save the mask vertices
    vertices=np.array(pts)
    mode_pts.append(vertices)

    # Note that the mask is applied on the STFT (not on the spectrogram)
    mode_rtf_warp = masque * tfr_w
    norm = 1 / NFFT / np.max(wind)
    mode_temp_warp = np.real(np.sum(mode_rtf_warp, axis=0)) * norm * 2

    # The modes are actually real quantities, so that they have negative frequencies.
    # Because we do not filter negative frequencies, we need to multiply by 2
    # the positive frequencies we have filter in order to recover the correct
    # modal energy
    mode = iwarp_temp_exa(mode_temp_warp, fs_w, r_1, c1_1, fs_1, N_ok)
    modes[:, i] = mode[:, 0]


    # Verification
    a = hilbert(mode)
    b = np.arange(1, N_ok + 1)
    h = np.hamming(N_window)
    mode_stft = tfrstft(a, b, NFFT, h)
    mode_spectro = abs(mode_stft) ** 2
    tm_1, D2 = momftfr(mode_spectro, 0, N_ok, time_ok)
    tm[:, i] = tm_1[:, 0]



print('End of filtering')

#--------------------------------------------------------------------------------------
## 7. Let's undo phase compensation

# Now we modify our estimated dispersion curve to undo phase compensation
# first we define the time-frequency law of our estimated source
f_source_est=IFLAW_est*fs
t_source_est=np.arange(0,len(IFLAW_est))/fs_1

# we need it on the same frequency axis than the dispersion curves
f = interpolate.interp1d(f_source_est[:,0], t_source_est[0,:], bounds_error=False, fill_value=np.nan )
t_source_est_ok = f(freq[:])

tm_est_with_source=np.zeros_like((tm))

for i in range(Nmodes):
    tm_est_with_source[:,i]=tm[:,i]+t_source_est_ok + t_dec2


#--------------------------------------------------------------------------------------
## 8. Verification

print('\n' * 20)
print('The red lines are the estimated dispersion curves.')
print('(on the right pannel, they are restricted to the frequency band you used for phase compensation)')

plt.figure()
plt.subplot(121)

if plot_in_dB == 1:
    plt.imshow(10 * np.log10(spectro), extent=[time_ok[0], time_ok[-1], freq[0], freq[-1]], aspect='auto', origin='low')
else:
    plt.imshow(spectro, extent=[time_ok[0], time_ok[-1], freq[0], freq[-1]], aspect='auto', origin='low')

plt.ylim([f_dec[0], f_dec[1]])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram and estimated dispersion curve')
for i in range(Nmodes):
    plt.plot(tm[:, i], freq[:], 'r')


plt.subplot(122)
if plot_in_dB==1:
    plt.imshow(10 * np.log10(spectro0), extent=[time0[0], time0[-1], freq[0], freq[-1]], aspect='auto', origin='low')
else:
    plt.imshow(spectro0, extent=[time0[0], time0[-1], freq[0], freq[-1]], aspect='auto', origin='low')

plt.ylim([f_dec[0], f_dec[1]])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title({'Original signal and estimated dispersion curves'})
for i in range(Nmodes):
    plt.plot(tm_est_with_source[:, i], freq[:], 'r')


plt.show(block=False)
input('Press ENTER to continue')

print('\n' * 20)
print('Let us restrict them further to a frequency band where they are ok')
print('You will have to enter the min/max frequency for each dispersion curves')
print('(for every mode, choose the widest frequency band over which the dispersion curve estimation looks correct)')



#--------------------------------------------------------------------------------------
## 9. Restrict to frequency band of interest

fmin = np.zeros((Nmodes))
fmax = np.zeros((Nmodes))

for i in range(Nmodes):
    fmin[i] = int(input('Enter min freq for mode ' + str(i + 1) + ':  '))
    fmax[i] = int(input('Enter max freq for mode ' + str(i + 1) + ':  '))

tm_ok = tm
tm_ok_s=tm_est_with_source
freq_sel = freq[:]

for i in range(Nmodes):
    pos = np.where((freq_sel > fmin[i]) & (freq_sel < fmax[i]))
    poss = np.array(pos, dtype='int')
    tm_ok[0:poss[0, 0], i] = np.nan
    tm_ok[poss[0, -1]:, i] = np.nan
    tm_ok_s[0:poss[0, 0], i] = np.nan
    tm_ok_s[poss[0, -1]:, i] = np.nan


print('\n' * 20)
print('This is the spectrogram with your best guess of the dispersion curves.')
print('')

plt.figure()
plt.subplot(121)

if plot_in_dB == 1:
    plt.imshow(10 * np.log10(spectro), extent=[time_ok[0], time_ok[-1], freq[0], freq[-1]], aspect='auto', origin='low')

else:
    plt.imshow(spectro, extent=[time_ok[0], time_ok[-1], freq[0], freq[-1]], aspect='auto', origin='low')

plt.ylim([f_dec[0], f_dec[1]])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram of signal after phase comp and estimated dispersion curves')
for i in range(Nmodes):
    plt.plot(tm_ok[:, i], freq[:], 'r')


plt.subplot(122)

if plot_in_dB==1:
    plt.imshow(10 * np.log10(spectro0), extent=[time0[0], time0[-1], freq[0], freq[-1]], aspect='auto', origin='low')

else:
    plt.imshow(spectro0, extent=[time0[0], time0[-1], freq[0], freq[-1]], aspect='auto', origin='low')


plt.ylim([f_dec[0], f_dec[1]])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title({'Original signal and estimated dispersion curves'})
for i in range(Nmodes):
    plt.plot(tm_ok_s[:, i], freq[:], 'r')



plt.show(block=False)
input('Press ENTER to continue')


#--------------------------------------------------------------------------------------
## 10. End

print('\n' * 20)
print('If you are happy with the result, the dispersion curves will be formatted and saved for localization')
print('If you are unhappy with the result, you will have to rerun the code')
save_data = input('Enter 1 to save the dispersion curve, or 0 to quit ')

if len(save_data) == 0:
    save_data = 0
else:
    save_data = int(save_data)

if save_data == 0:
    print(' ')
    print('Try again !')
    print(' ')
    print('END')

else:
    fmin_data = np.min(fmin)
    fmax_data = np.max(fmax)
    print('\n' * 20)
    print(['The estimated dispersion curves go from fmin=' + str(fmin_data) + ' to fmax=' + str(
        fmax_data) + ' Hz and will be saved only between these 2 values'])
    print('We now need to choose a frequency spacing df, so that the data will be stored along the frequency axis fmin:df:fmax')
    print('Small values of df are good since it provides more information for inversion, but it also slow down the process')
    print('A good rule of thumb is to have 50-100 points between fmin and fmax')
    df_data = input('Enter the df value you want (df=2 is ok for the gunshot provided in the tutorial) ')

    if len(df_data) == 0:
        df_data = 2
    else:
        df_data = int(df_data)

    freq_data = np.arange(fmin_data, fmax_data, df_data)
    Nf_data = len(freq_data)

    data = np.zeros((Nf_data, Nmodes))
    data[:] = np.nan

    for mm in np.arange(0, Nmodes):
        f = interpolate.interp1d(freq, tm_ok[:, mm], bounds_error=False, fill_value=np.nan)
        data[:, mm] = f(freq_data)

    today = date.today()
    output_file = 'upsweep_modes_' + str(today) + '.mat'

    sio.savemat(output_file, {'data': data, 'freq_data': freq_data, 'Nmode': Nmodes, 'time_ok': time_ok,
                              'freq': freq, 'spectro': spectro, 'modes': modes, 'fs': fs_1, 'tm_ok': tm_ok, 's_ok': s_ok_1,
                              't_dec': t_dec, 'r': r, 'c1': c1, 'masque': masque, 'time_w': time_w, 'freq_w': freq_w,
                              'spectro_w': spectro_w, 'fmax_plot': fmax_plot, 'fmin': fmin, 'fmax': fmax,
                              'mode_pts': mode_pts})

    print('\n' * 20)
    print('The data have been saved in a mat file with the same name than the')
    print('original wavefile, follow by the current date and time')
    print('The mat file is located in the same folder than the original wav file')
    print('You can now proceed with localization')
    print(' ')
    print('END')

