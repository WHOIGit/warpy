# Warping tutorial
# optional_create_simulated_signal

# April 2020
# Eva Chamorro - Daniel Zitterbart - Julien Bonnel

#--------------------------------------------------------------------------------------
## 1. Import packages
import numpy as np
import scipy.io as sio
from scipy.fftpack import ifft
from subroutines.pekeris import *
import warnings
warnings.filterwarnings('ignore')

#--------------------------------------------------------------------------------------
## 2. Environment

# Waveguide parameters
rho1 = 1000  ### density in water (kg/m^3)
rho2 = 1500  ### density in seabed (kg/m^3)
c1 = 1500  ###  sound speed in water (m/s)
c2 = 1600  ### sound speed in seabed (m/s)
D = 100  ### water depth (m)

r = 10000  ### source/receiver range

fmax = 100  ### max frequency for simulation
fs = 2 * fmax  ### sampling frequency is twice fmax
df = 0.1
nfft = fs / df  ### make sure that nfft is an integer by choosing appropriate df

if nfft != np.floor(nfft):
    raise ValueError('make sure that nfft is an integer by choosing an appropriate df')

freq_env = np.arange(0, fmax + df, df)  ### freq axis to simulate propagation
freq_sig = np.arange(0, nfft) * df  ### freq axis for simulated signal, obtained after Fourier synthesis

# Compute wavenumbers and normalization coefficients
[kr, kz, A2] = pek_init(rho1, rho2, c1, c2, D, freq_env)

# Compute Green's function
zs = D  # source depth (m) - single source only (only one value)
zr = np.arange(0, D + 1)  # receiver depths (m) - can be a vertical array (vector)
zr = zr[np.newaxis, :]

# Green's function (frequency domain), dimension [Nb freq , Nb capt]
g = pek_green(kr, kz, A2, zs, zr, r)

#--------------------------------------------------------------------------------------
## 3. Build signal at a given depth

z_sig=D
ind_z=np.argmin(np.abs(z_sig-zr))
s_f=np.zeros(len(freq_sig),'complex')
s_f[0:len(freq_env)]=g[:,ind_z]


# Go to time domain
s_f=s_f[np.newaxis,:]
s_t=np.real(ifft(s_f))#symetric
norm_s_t=np.max(np.abs(s_t))
s_t=s_t/norm_s_t

# Time shift the signal by r/c1
t_dec=r/c1
i=complex(0,1)
s_f_dec=s_f*np.exp(2*1*i*np.pi*freq_sig*t_dec)

# Go to time domain
s_t_dec=ifft(s_f_dec) # 'symmetric'

norm_s_t=np.max(np.abs(s_t_dec))
s_t_dec=s_t_dec/norm_s_t

zr=z_sig

sio.savemat('sig_pek_for_warp.mat',{'s_t': s_t, 's_t_dec': s_t_dec,'rho1':rho1,'rho2':rho2,
                        'c1':c1,'c2':c2,' D': D,'fs':fs,'r':r,'zs':zs,'zr':zr })


#--------------------------------------------------------------------------------------
## 4. Separated modes


(Nf, Nm) = np.shape(kr)

mod = np.zeros((Nf, Nm), 'complex')

for mm in np.arange(0, Nm):
    mod[:, mm] = A2[:, mm] * np.sin(kz[:, mm] * zs) * np.exp(-1 * i * kr[:, mm] * r) / np.sqrt(kr[:, mm] * r) * np.sin(
        kz[:, mm] * zr)

mod[kr == 0] = 0

mode_f = np.zeros((len(freq_sig), Nm), 'complex')
mode_f[0:len(freq_env), :] = mod
mode_t = np.real(ifft(mode_f, len(freq_sig), axis=0))  # 'symmetric'
mode_t = mode_t / norm_s_t

[vg, f_vg] = pek_vg(0, fmax, 1, Nm, c1, c2, rho1, rho2, D, df)

sio.savemat('sig_pek_and_modes_for_warp.mat', {'s_t': s_t, 'rho1': rho1, 'rho2': rho2, 'c1': c1,
                                               'c2': c2, ' D': D, 'fs': fs, 'r': r, 'zs': zs, 'zr': zr,
                                               'mode_t': mode_t, 'vg': vg, 'f_vg': f_vg})

print('')
print('END')

