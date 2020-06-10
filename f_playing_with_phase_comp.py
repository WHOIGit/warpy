# Warping tutorial
## f_playing_with_phase_comp

##### May 2020
###### Eva Chamorro - Daniel Zitterbart - Julien Bonnel

## 1. Import packages

import os
import matplotlib

matplotlib.use('TkAgg')
import scipy.io as sio
from matplotlib import interactive
from scipy import interpolate
from scipy.fftpack import ifft
from scipy.signal import hilbert
from functions.warping_functions import *
from functions.time_frequency_analysis_functions import *
from functions.filter import *



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



#--------------------------------------------------------------------------------------
## 3. Process signal

# The first sample of s_t_dec corresponds to time r/c1
# Make the signal shorter, no need to keep samples with zeros
N_ok=250
ir_ok=s_t_dec[:,0:N_ok] ### this is the impulse response of the waveguide


#--------------------------------------------------------------------------------------
## 4. Source signal creation

X,IFLAW_source=fmpar(200,np.array([1,0.5]),np.array([100,0.15]),np.array([200,0.01]))


# source signal is a parabolic modulated FM signal
source=X*(IFLAW_source+1)
N_source=len(source)

#--------------------------------------------------------------------------------------
## 5. Propagated signal

# Propagated signal = time-convolution between source and impulse response = multiplication in the frequency domain
source_f=fft(source, N_ok)
source_f=source_f[:,np.newaxis]
ir_f=fft(ir_ok,N_ok)
s_f=source_f*np.transpose(ir_f)
s_ok=ifft(s_f,N_ok, axis=0) #propagated signal

# STFT computation
NFFT=1024
N_window=31 # you need a short window to see the modes
b=np.arange(1,N_ok+1)
b=b[np.newaxis,:]
d=np.hamming(N_window)
d=d[:,np.newaxis]
tfr=tfrstft(s_ok,b,NFFT,d)
source_1=source[:,np.newaxis]
tfr_source=tfrstft(source_1,b,NFFT,d)

spectro=abs(tfr)**2
spectro_source=abs(tfr_source)**2


# Time and frequency axis of the original signal
time=np.arange(0,N_ok)/fs
freq=np.arange(0,NFFT)*fs/NFFT


time_init=time
freq_init=freq
spectro_init=spectro

#--------------------------------------------------------------------------------------
## 6 Phase compensation

flag_pc=0

while flag_pc ==0:

    print('This is the same example as has been shown previously (see e_warping_and_filtering_and_phase_comp.m)')
    print('This time, we will approximate the source TF law with 3 linear pieces.')
    print('As explained in Sec V C, an easy way to do so is to roughly follow the TF contour of mode 1')
    print('Let us try')
    print('')

    print('Click 4 times on the spectrogram to define your 3 linear pieces')
    print('In this case, the easiest is probably to start at early-times/high frequencies,')
    print('and to progress toward increasing times and decreasing frequencies')
    print('(roughly follow the black line from the previous example)')

    plt.figure()
    plt.imshow(spectro, extent=[time[0, 0], time[0, -1], freq[0, 0], freq[0, -1]], aspect='auto', origin='low')
    plt.ylim([0, fs / 2])
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Received signal')
    plt.show(block=False)

    interactive(True)
    pts = plt.ginput(4)
    input('Pres ENTER to continue and plot the result')

    c = np.array(pts)
    ttt = c[:, 0]
    fff = c[:, 1]

    print('\n' * 20)
    print('The black line is your best guess of the source signal')
    print('Now, let us do phase compensation to transform our received signal')
    print('into something that looks like the impulse response of the waveguide')
    print('Close the figure to continue')

    ### let's plot it on top of the received signal
    plt.figure()
    plt.imshow(spectro, extent=[time[0, 0], time[0, -1], freq[0, 0], freq[0, -1]], aspect='auto', origin='low')
    plt.ylim([0, fs / 2])
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Received signal')
    plt.plot(ttt, fff, 'black', linewidth=3)
    plt.show(block='True')

    ### convert to samples and reduced frequencies
    ttt_s = np.round(ttt * fs) + 1
    fff_s = fff / fs
    n_click = 4

    ### create the piecewise linear-FM source
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


    ## Phase correction
    sig_prop_f = fft(s_ok, axis=0)
    i = complex(0, 1)
    sig_rec_f = sig_prop_f * np.exp(-i * phi)  # note that only source phase is deconvoluted
    sig_rec_t = ifft(sig_rec_f, axis=0)  # Signal in time domain after source deconvolution

    # Figure
    b = np.arange(1, N_ok + 1)
    b = b[np.newaxis, :]
    d = np.hamming(N_window)
    d = d[:, np.newaxis]

    tfr_sig_rec = tfrstft(sig_rec_t, b, NFFT, d)

    print('\n' * 20)
    print('Here is the result')
    print('Close the figure to continue')

    plt.figure()
    plt.imshow(abs(tfr_sig_rec) ** 2, extent=[time[0, 0], time[0, -1], freq[0, 0], freq[0, -1]], aspect='auto',
               origin='low')
    plt.ylim([0, fs / 2])
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Signal after phase compensation')
    plt.show(block=True)

    print('If you think it looks like an impulse response, press 1 to continue')
    print('If you want to redo the phase compensation, press 0  ')
    flag_pc = input('(you will have the opportunity to modify the time origin later ')

    if flag_pc == '' or  flag_pc == '1':
        flag_pc = 1

    elif flag_pc == '0':
        flag_pc = int( flag_pc)




#--------------------------------------------------------------------------------------
## 7 Play with time origin


redo_dt = 0
while redo_dt == 0:

    print('\n' * 20)
    print('You must now define the time origin')
    print('Double-click once on the spectrogram at the position where you want to define the time origin')

    plt.figure()
    plt.imshow(np.abs(tfr_sig_rec)** 2, extent=[time[0,0], time[0,-1], freq[0,0], freq[0,-1]],aspect='auto',
               origin='low' )
    plt.ylim([0,fs / 2])
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Signal after phase compensation')
    plt.show(block=False)

    interactive(True)
    t_dec = plt.ginput(1)
    input('Pres ENTER to continue and plot the result')
    t_dec = t_dec[0]
    t_dec = t_dec[0]

    ## Shorten the signal, make it start at the chosen time origin, and warp it
    time_t_dec = np.abs(time - t_dec)
    ind0 = np.where(time_t_dec == np.min(time_t_dec))
    ind0 = ind0[1]
    ind0 = ind0[0]

    s_ok = sig_rec_t[ind0:]
    N_ok = len(s_ok)

    # Corresponding time and frequency axis
    time_ok = np.arange(0, N_ok) / fs

    # The warped signal will be s_w
    s_ok_1 = np.transpose(s_ok)
    s_w, fs_w = warp_temp_exa(s_ok_1, fs, r, c1)
    M = len(s_w)

    ### Original signal
    N_window = 31  # you need a short window to see the modes
    a = np.arange(1, N_ok + 1)
    a = a[np.newaxis, :]
    h = np.hamming(N_window)
    h = h[:, np.newaxis]
    tfr = tfrstft(s_ok, a, NFFT, h)
    spectro = abs(tfr) ** 2

    ### Warped signal
    N_window_w = 301  # You need a long window to see the warped modes
    wind = np.hamming(N_window_w)
    wind = wind / np.linalg.norm(wind)
    wind = wind[:, np.newaxis]
    b = np.arange(1, M + 1)
    b = b[np.newaxis, :]
    tfr_w = tfrstft(s_w, b, NFFT, wind)
    spectro_w = abs(tfr_w) ** 2

    # Time and frequency axis of the warped signal
    time_w = np.arange(0, (M) / fs_w, 1 / fs_w)
    freq_w = np.arange(0, fs_w - fs_w / NFFT + fs_w / NFFT, fs_w / NFFT)

    # Figure

    print('\n' * 20)
    print('The left panel shows the spectrogram of the original ')
    print('signal with the chosen time origin')
    print('The right panel shows the corresponding warped signal')
    print('Close the figure to continue')
    print('')


    plt.figure()
    plt.subplot(121)
    plt.imshow(spectro, extent=[time_ok[0, 0], time_ok[0, -1], freq[0, 0], freq[0, -1]], aspect='auto', origin='low')
    plt.ylim([0, fs / 2])
    plt.xlim([0, 0.5])  ### Adjust this to see better
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Original signal with chosen time origin')

    plt.subplot(122)
    plt.imshow(spectro_w, extent=[time_w[0], time_w[-1], freq_w[0], freq_w[-1]], aspect='auto', origin='low')
    plt.ylim([0, 40])  ### Adjust this to see better
    plt.xlabel('Warped time (sec)')
    plt.ylabel('Corresponding warped frequency (Hz)')
    plt.title('Corresponding warped signal')
    plt.show(block=True)


    print('Type 0 if you want to redo the time origin selection')
    redo_dt = input('or type 1 if you want to proceed with modal filtering ')

    if redo_dt == '' or redo_dt == '1':
        redo_dt = 1

    elif redo_dt == '0':
        redo_dt = int(redo_dt)


#--------------------------------------------------------------------------------------
## 8. Filtering

## selection the spectro_w part to mask
spectro_w_1=spectro_w[0:200,:]
spectro_w_1=np.transpose(spectro_w_1)
time_w_1=time_w
freq_w_1=freq_w[:200]
Nmode=4

modes=np.zeros((N_ok,Nmode))
tm=np.zeros((NFFT,Nmode))



def pol(arr):

    ## create GUI
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

    ## Start Qt event loop unless running in interactive mode or using pyside.
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

    ## Verification

    a = hilbert(mode)
    b = np.arange(1, N_ok + 1)
    b = b[np.newaxis, :]
    h = np.hamming(N_window)
    h = h[:, np.newaxis]
    mode_stft = tfrstft(a, b, NFFT, h)
    mode_spectro = abs(mode_stft) ** 2
    tm_1, D2 = momftfr(mode_spectro, 0, N_ok, time_ok)
    tm[:, i] = tm_1[:, 0]
    if i < Nmode:
        print('Filtering mode '+ str(i+2))


print('End of filtering')


## 9. Verification

print('\n' * 20)
print('The red lines are the estimated dispersion curves.')
print('For real life applications, you will have to restrict them to a frequency band where they are ok')
print('Now we need to undo the phase compensation to look at the true filtered mode')
print('Close the figure to continue to look at your result vs the true modes')

plt.figure()
plt.imshow(spectro, extent=[time_ok[0,0], time_ok[0,-1], freq[0,0], freq[0,-1]],aspect='auto',origin='low')
plt.ylim([0,fs/2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram and estimated dispersion curve')

for i in range(Nmode):
    plt.plot(tm[:,i],freq[0, :],'r')

plt.show(block=True)



## 10. Let's undo phase compensation

### look at e_warping_and_filtering_and_phase_comp.m if you want to undo
### phase compensation for the filtered modes. Here, we will only undo it
### for the dispersion curves

data = sio.loadmat(os.getcwd()+ '/sig_pek_and_modes_for_warp.mat')
r=data['r']
vg=data['vg']
c1=data['c1']
f_vg=data['f_vg']

#### we need a few tricks to create the theoretical dispersion curves
###### first the dispersion curves of the impulse response

tm_theo_ir=r/vg-r/c1 ### range over group_speed minus correction for time origin

##### now we need to include the source law ....
f_source=IFLAW_source*fs
t_source=np.arange(0,len(IFLAW_source))/fs

f = interpolate.interp1d(f_source[0,:], t_source[0,:], bounds_error=False, fill_value=np.nan )
t_source_ok = f(f_vg[0,:])
t_source_ok_1=np.tile(t_source_ok,(5,1))
tm_theo_with_source=tm_theo_ir+t_source_ok_1


#### Now we modify our estimated dispersion curve to undo phase compensation
###### first we define the time-frequency law of our estimated source

f_source_est=IFLAW_est*fs
t_source_est=np.arange(0,len(IFLAW_est))/fs

##### we need it on the same frequency axis than the dispersion curves

f = interpolate.interp1d(f_source_est[:,0], t_source_est[0,:], bounds_error=False, fill_value=np.nan )
t_source_est_ok = f(freq[0,:])
t_source_est_ok=t_source_est_ok[:,np.newaxis]
t_source_est_ok_1=np.tile(t_source_ok,(4,1))
tm_est_with_source=tm+t_source_est_ok+t_dec

print('\n' * 20)
print('This is the spectrogram of the original signal,')
print('the true dispersion curves (black) and your estimation (red).')
print('Remember that the estimated dispersion curve are ')
print('relevant only in the frequency band where they match')
print('the estimated spectrogram.')
print('If you do not like the result, try to redo phase compensation and/or change time origin')
print('Close the figure to exit the code')


plt.figure()
plt.imshow(spectro_init, extent=[time_init[0,0], time_init[0,-1], freq_init[0,0], freq_init[0,-1]],aspect='auto', origin='low' )
plt.xlim([0,1.2])
plt.ylim([0,fs/2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Original signal and estimated dispersion curve')


for i in range(Nmode):
    plt.plot(tm_theo_with_source[i,:], f_vg[0,:], 'black')

    plt.plot(tm_est_with_source[:,i],freq[0,:], linewidth=2, color='r')

plt.show(block=True)

print(' ')
print('END')
