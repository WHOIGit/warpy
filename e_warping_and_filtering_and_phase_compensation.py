# Warping tutorial
# e_warping_and_filtering_and_phase_comp

# May 2020
# Eva Chamorro - Daniel Zitterbart - Julien Bonnel

#--------------------------------------------------------------------------------------
## 1. Import packages
import os
import scipy.io as sio
from scipy.signal import hilbert
from scipy import interpolate
from scipy.fftpack import ifft
from subroutines.warping_functions import *
from subroutines.time_frequency_analysis_functions import *
from subroutines.filter import *

import warnings
warnings.filterwarnings('ignore')

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



#--------------------------------------------------------------------------------------
## 3. Process signal

# The first sample of s_t_dec corresponds to time r/c1
# Make the signal shorter, no need to keep samples with zeros
N_ok=250
ir_ok=s_t_dec[:,0:N_ok]



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
ir_f=fft(ir_ok,N_ok)
s_f=np.transpose(source_f*ir_f)
s_ok=ifft(s_f,N_ok, axis=0) #propagated signal

# STFT computation
NFFT=1024
N_window=31 # you need a short window to see the modes
b=np.arange(1,N_ok+1)
d=np.hamming(N_window)
tfr=tfrstft(s_ok,b,NFFT,d)
tfr_source=tfrstft(source,b,NFFT,d)

spectro=abs(tfr)**2
spectro_source=abs(tfr_source)**2

# Time and frequency axis of the original signal
time=np.arange(0,N_ok)/fs
freq=np.arange(0,NFFT)*fs/NFFT

print('The source is a non-linear (parabolic) FM signal')
print('The received signal is highly influenced by the source')
input('Press ENTER to continue')
print('\n' * 30)
print('Looking at the received signal, you may think that the source is a linear FM downsweep.')
print('Although wrong, this is a good enough approximation for warping')
print('Let us illustrate this, close the figure to continue')
print('')


# Figure
plt.figure(figsize=(10,7))
plt.subplot(121)
plt.imshow(spectro_source, extent=[time[0,0], time[0,-1], freq[0,0], freq[0,-1]],aspect='auto', origin='low')
plt.ylim([0, fs/2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Source signal')

plt.subplot(122)
plt.imshow(spectro,extent=[time[0,0], time[0,-1], freq[0,0], freq[0,-1]],aspect='auto', origin='low')
plt.ylim([0, fs/2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Received signal')
plt.show(block=False)
input('Pres ENTER to continue')



#--------------------------------------------------------------------------------------
## 6. Phase compensation

# The guess source will be a linear sweep with following parameters
L=130   ### length of the sweep in samples
f1=0.5  ### start frequency of the sweep (reduced frequency, i.e. f1=0.5*fs Hz)
f2=0.01 ### end frequency of the sweep (reduced frequency, i.e. f2=0.01*fs Hz)

source_est, IFLAW_est=fmlin(L,f1,f2) ### estimated source signal in the time domain
source_est_f=fft(source_est,N_ok,axis=0) ### estimated source signal in the frequency domain
phi=np.angle(source_est_f)                 ### phase of the estimated source signal in the frequency domain


# let's plot it on top of the received signal
print('\n' * 30)
print('As an example, the black line may be our best guess of the source signal')
print('Now, let us do phase compensation to transform our received signal')
print('into something that looks like the impulse response of the waveguide')

# Figure
plt.figure(figsize=(7,5))
plt.subplot(121)
plt.imshow(spectro_source, extent=[time[0,0], time[0,-1], freq[0,0], freq[0,-1]],aspect='auto', origin='low')
plt.ylim([0, fs/2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Source signal')

# Figure
plt.subplot(122)
plt.imshow(spectro, extent=[time[0,0], time[0,-1], freq[0,0], freq[0,-1]],aspect='auto', origin='low')
plt.ylim([0, fs/2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Received signal')
x1=0
x2=(L-1)/fs
y1=0.5*fs
y2=0.01*fs
plt.plot([0, x2[0,0]], [y1[0,0],y2[0,0]], linewidth=3, color='black')
plt.show(block=False)
input('Pres ENTER to continue')


# 6.1 Phase correction

sig_prop_f=fft(s_ok,axis=0)
i=complex(0,1)
sig_rec_f=sig_prop_f*np.exp(-i*phi) # note that only source phase is deconvoluted
sig_rec_t=ifft(sig_rec_f,axis=0)  #Signal in time domain after source deconvolution


# Figure
b=np.arange(1,N_ok+1)
d=np.hamming(N_window)
tfr_sig_rec=tfrstft(sig_rec_t,b,NFFT,d)

print('\n' * 30)
print('The result is not too bad, it indeed looks like an impulse response with modes')
print('Now let us warp!')

plt.figure()
plt.imshow(abs(tfr_sig_rec)**2, extent=[time[0,0], time[0,-1], freq[0,0], freq[0,-1]],aspect='auto', origin='low')
plt.ylim([0,fs/2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Signal after phase compensation')
plt.show(block=False)
input('Pres ENTER to continue')


#--------------------------------------------------------------------------------------
## 7. Warping

# The warped signal will be s_w

s_w, fs_w=warp_temp_exa(np.transpose(sig_rec_t),fs,r,c1)
M=len(s_w)

# STFT computation
N_window_w=301 # You need a long window to see the warped modes
wind=np.hamming(N_window_w)/np.linalg.norm(np.hamming(N_window_w))
b=np.arange(1,M+1)
tfr_w=tfrstft(s_w,b,NFFT,wind)
spectro_w=abs(tfr_w)**2

# Time and frequency axis of the warped signal
time_w=np.arange(0,M)/fs_w
freq_w=np.arange(0,NFFT)*fs_w/NFFT # this is actually not exact, but precise enough for a rough example





#--------------------------------------------------------------------------------------
## 8. Filtering

print('\n' * 30)
print('Not too bad huh?')
print('Now click on the spectrogram to create the mask to filter one mode.')
print('(see c_warping_and_filtering.m if necessary)')
print(' ')


# WARNING "IF I DELETE THE FUNCTION pol FROM THE SCRIPT IS NOT WORKING CORRECTLY" ######
# func ########

def pol(arr):

    # create GUI
    app = QtGui.QApplication([])
    w = pg.GraphicsWindow(size=(1000, 800), border=True)
    w.setWindowTitle('pyqtgraph: Filtering')

    text = """Filtering: filter a single mode .<br>\n
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

    # Start Qt event loop unless running in interactive mode or using pyside.
    if __name__ == '__main__':
        import sys

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    mask = roi.getArrayRegion(arr, img1a)
    pts = roi.getState()['points']

    return mask, pts


# selection the spectro_w part to mask
spectro_w_1=spectro_w[0:200,0:648]
spectro_w_1=np.transpose(spectro_w_1)

# Create the mask
maska, pts = pol(spectro_w_1)
final_mask = maskc(maska, pts,spectro_w_1 )
mask_ini=np.transpose(final_mask)

# add the part masked to the total sprectogram array
masque_2=np.zeros_like(spectro_w[200:,:])
masque_3=np.zeros_like(spectro_w[0:200,648:])
masque=np.concatenate((mask_ini,masque_3),axis=1)
masque=np.concatenate((masque,masque_2),axis=0)

# Note that the mask is applied on the STFT (not on the spectrogram)
mode_rtf_warp=masque*tfr_w
norm=1/NFFT/np.max(wind)
mode_temp_warp=np.real(np.sum(mode_rtf_warp,axis=0))*norm
mode=iwarp_temp_exa(mode_temp_warp,fs_w,r,c1,fs,N_ok)


#--------------------------------------------------------------------------------------
## 9. Verification

# you can estimate the dispersion curve by computing the
# frequency moment of the filtered mode TFR
a=hilbert(mode)
b=np.arange(1,N_ok+1)
h=np.hamming(N_window)
mode_stft=tfrstft(a,b,NFFT,h)
mode_spectro=np.abs(mode_stft)**2
tm,D2=momftfr(mode_spectro,0,N_ok,time)

print('\n' * 30)
print('The red line is the estimated dispersion curve.')
print('You have to restrict it to a frequency band where it is relevant')
print('If the result is not satisfying, you must create a new filtering mask')
print('Let us assume that the result is ok.')
print('Now we need to undo the phase compensation to look at the true filtered mode')

# Figure
plt.figure()
plt.imshow(abs(tfr_sig_rec)**2, extent=[time[0,0], time[0,-1], freq[0,0], freq[0,-1]],aspect='auto',origin='low')
plt.ylim([0,fs/2])
plt.xlim([0,1])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Signal after phase compensation and estimated dispersion curve')

plt.plot(tm[:,0],freq[0,:], linewidth=2, color='r')
plt.show(block=False)
input('Pres ENTER to continue')



#--------------------------------------------------------------------------------------
## 10. Let's undo phase compensation

mode_f=fft(mode,axis=0)
i=complex(0,1)
mode_rec_f=mode_f*np.exp(i*phi)
mode_rec_t=ifft(mode_rec_f,axis=0) ### Mode in time domain after source re-convolution
b=np.arange(1,N_ok+1)
d=np.hamming(N_window)
mode_rec_sp=tfrstft(mode_rec_t,b,NFFT,d)
mode_rec_sp=abs(mode_rec_sp)


# We need to undo the source deconvolution for the dispersion curve too
# first we define the time-frequency law of our estimated source
f_source_est=IFLAW_est*fs
t_source_est=np.arange(0,len(IFLAW_est))/fs


# we need it on the same frequency axis than the dispersion curves
f = interpolate.interp1d(f_source_est[:,0], t_source_est[0,:], bounds_error=False, fill_value=np.nan )
t_source_est_ok = f(freq[0,:])
tm_est_with_source=tm[:,0]+t_source_est_ok

print('\n' * 30)
print('The left subplot is the spectrogram of the filtered mode')
print('The right subplot shows the received signal and the estimated dispersion curve')
print('Now, let us compare your result with the true modes')
print('')


plt.figure(figsize=(7,5))
plt.subplot(121)
plt.imshow(mode_rec_sp, extent=[time[0,0], time[0,-1], freq[0,0], freq[0,-1]],aspect='auto',origin='low')
plt.ylim([0, fs/2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Filtered mode')

plt.subplot(122)
plt.imshow(spectro, extent=[time[0,0], time[0,-1], freq[0,0], freq[0,-1]],aspect='auto',origin='low')
plt.ylim([0, fs/2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Original signal and estimated dispersion curve')
plt.plot(tm_est_with_source,freq[0,:],linewidth=3, color='black')
plt.show(block=False)
input('Pres ENTER to continue')


#--------------------------------------------------------------------------------------
## 11. Verification

data = sio.loadmat(os.getcwd()+ '/sig_pek_and_modes_for_warp.mat')
r=data['r']
vg=data['vg']
c1=data['c1']
f_vg=data['f_vg']

# we need a few tricks to create the theoretical dispersion curves
# first the dispersion curves of the impulse response

tm_theo_ir=r/vg-r/c1 ### range over group_speed minus correction for time origin
# now we need to include the source law ....
f_source=IFLAW_source*fs

t_source=np.arange(0,len(IFLAW_source))/fs


# but we need it on the same frequency axis than tm_theo_ir
f = interpolate.interp1d(f_source[0,:], t_source[0,:], bounds_error=False, fill_value=np.nan )
t_source_ok = f(f_vg[0,:])
t_source_ok_1=np.tile(t_source_ok,(5,1))
tm_theo_with_source=tm_theo_ir+t_source_ok_1


print('\n' * 30)
print('This is the same figure than before, except that')
print('the true dispersion curves are now in black.')
print('Remember that the estimated dispersion curve is ')
print('relevant only in the frequency band where it matches')
print('the estimated spectrogram.')
print(' ')


plt.figure()
plt.imshow(spectro, extent=[time[0,0], time[0,-1], freq[0,0], freq[0,-1]],aspect='auto',origin='low')
plt.ylim([0,fs/2])
plt.xlim([0,1.2])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Original signal and estimated dispersion curve')
plt.plot(tm_theo_with_source[0,:], f_vg[0,:], 'black')
plt.plot(tm_theo_with_source[1,:], f_vg[0,:], 'black')
plt.plot(tm_theo_with_source[2,:], f_vg[0,:], 'black')
plt.plot(tm_theo_with_source[3,:], f_vg[0,:], 'black')
plt.plot(tm_est_with_source,freq[0,:], linewidth=3, color='r')
plt.show(block=False)
input('Pres ENTER to continue and exit the code')




print('END')





