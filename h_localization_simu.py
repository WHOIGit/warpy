# Warping tutorial
# h_localization_simu

# May 2020
# Eva Chamorro - Daniel Zitterbart - Julien Bonnel


#--------------------------------------------------------------------------------------
## 1. Import packages

import os
import scipy.io as sio
import numpy as np
import cmath
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from subroutines.warping_functions import *
from subroutines.time_frequency_analysis_functions import *
from subroutines.pekeris import *
from subroutines.hamilton import *

import warnings
warnings.filterwarnings('ignore')

'NOTE: This code has to be run in the Terminal'

#--------------------------------------------------------------------------------------
## 2. Load data

data_up = sio.loadmat(os.getcwd()+ '/data_for_loc.mat')

# Select variables
data=data_up['data']
freq_data=data_up['freq_data']
c1_true=data_up['c1_true']
c2_true=data_up['c2_true']
D_true=data_up['D_true']
rho1_true=data_up['rho1_true']
r_true=data_up['r_true']
dt_true=data_up['dt_true']

Nm=len(data[0,:]) ## number of modes
Nf=len(freq_data[0,:])

print('These are estimated dispersion curves obtained with')
print('the previous code (g_warping_filtering_multiple_modes.m)')
print('They will be used as data for localization')
print('')


plt.figure()
for i in range(Nm):
    plt.plot(data[:,i], freq_data[0,:],'black')

plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.grid()
plt.title('Dispersion curves')
plt.show(block=False)
input('Pres ENTER to continue ')

#--------------------------------------------------------------------------------------
## 3. Compute replicas

# We need to know some of the environmental parameters
c1 = c1_true[0, 0]
D = D_true[0, 0]
rho1 = rho1_true[0, 0]

# We will look for range, seabed sound speed, and time shift on the
# following grids
r_ = np.arange(5000, 15100, 100)
c2_ = np.arange(1550, 1710, 10)
dt_ = np.arange(-7, -5.995, 0.005)

# NB: dt values should be roughly between rmin/c1 and rmax/c1. One can start with wide
#    search bounds and coarse steps for dt, and gradually narrow the bounds and decrease the steps.

Nr = len(r_)
Nc = len(c2_)
Nt = len(dt_)

fmin = freq_data[0, 0]
fmax = freq_data[0, -1]
df = freq_data[0, 1] - freq_data[0, 0]

vg = np.zeros((Nc, Nm, Nf))

m_min = 1
m_max = Nm

print('\n' * 20)
print('The first step is to compute replicas')
print('Computing replicas ....')

for cc in (np.arange(0, Nc)):
    c2 = c2_[cc]
    rho2 = hamilton(c2) * 1000
    [vg[cc, :, :], f_rep] = pek_vg(fmin, fmax, m_min, m_max, c1, c2, rho1, rho2, D, df)

print('Replicas computed!')
input('Press ENTER to continue')

#--------------------------------------------------------------------------------------
## 4. Localization: impulsive sources
print('\n' * 20)
print('First, let us localize the source with Eq. (19),')
print('i.e. we know the source is an impulse')
input('Press ENTER to proceed')

print('\n' * 20)
print('Starting localization')

J = np.zeros((Nt, Nr, Nc))
Nsteps = Nt * Nr
step = 1

for tt in (np.arange(0, Nt)):

    for rr in (np.arange(0, Nr)):
        # waitbar(step/Nsteps,www,'Localization in progress ...');
        r = r_[rr]
        for cc in (np.arange(0, Nc)):
            rep = np.transpose(r / np.squeeze(vg[cc, :, :]) + dt_[tt])
            J[tt, rr, cc] = np.nanmean(np.nanmean(abs(data - rep) ** 2))
        # end
        # step=step+1;

v = np.min(J[:, :, :])
tt_m, rr_m, cc_m = np.where(J == v)

dt_est = dt_[tt_m]
r_est = r_[rr_m]
c2_est = c2_[cc_m]

print('Localization done!')
print('')
print('')
print('Estimated range: ' + str(r_est[0]) + ' m')
print('Estimated time shift: ' + str(dt_est[0]) + ' s')
print('Estimated seabed sound speed: ' + str(c2_est[0]) + ' m/s')
print('True values are: range ' + str(r_true[0, 0]) + ' m ; time shift ' + str(dt_true[0, 0]) + ' s ; speed ' + str(
    c2_true[0, 0]) + ' m/s')

input('Press ENTER to plot results')


# 4.1 Plot results

rep_est=r_est/np.squeeze(vg[cc_m,:,:])+dt_est

print('\n' * 20)
print('The top panel of the figure shows the data (in black) and the predicted dispersion curves')
print('(i.e. the best replicas) in black. If there is not a good match')
print('between the 2, then the localization is likely wrong')
print(' ')
print('The bottom panel of the figure shows least square fit for range')
print('If the curve is not relatively smooth with a marked minimum, then')
print('the localization is likely wrong.')


plt.figure()
plt.subplot(211)
for i in range(Nm):

    plt.plot(data[:,i], freq_data[0,:],'black')

    plt.plot(rep_est[i,:], freq_data[0,:],'or',fillstyle='none')

plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Dispersion curves')
plt.grid()
plt.subplot(212)
plt.plot(r_/1000,np.squeeze(J[tt_m,:,cc_m]))
plt.grid()
plt.xlabel('Range (km)')
plt.title('Least square fit')

plt.show(block=False)
input('Pres ENTER to continue')

#--------------------------------------------------------------------------------------
## 5. Localization: non-impulsive sources

# Localization: non-impulsive sources
print('\n' * 20)
print('Now, let us localize the source with Eq. (20),')
print('i.e. we do not know if the source is an impulse')
print('')


Nm_inv = 4
J = np.zeros((Nr, Nc, Nm_inv - 1))

# we need common frequencies between modes. Let us verify the data
flag_data = 0
for mm in np.arange(0, Nm_inv - 1):

    data_ok = data[:, mm + 1] - data[:, mm]
    na = np.where(~np.isnan(data_ok))

    if np.size(na) != 0:
        flag_data = 1

if flag_data == 0:  ### there are no common frequency band between modes

    print('There are no common frequency band between modes')
    print('It is impossible to use Eq. (29)')
    print('Go back to the previous code, and redo the dispersion curve estimation')
    print('Make sure to have an overlap in frequency between the modes')
    print(' ')
    print('END')

else:  ## there are common frequencies and we can proceed

    for rr in np.arange(0, Nr):
        r = r_[rr]
        # waitbar(rr/Nr,www,'Localization in progress ...');
        for cc in np.arange(0, Nc):

            for mm in np.arange(0, Nm_inv - 1):
                rep = (r / np.squeeze(vg[cc, mm + 1, :]) - r / np.squeeze(vg[cc, mm, :]))
                data_ok = data[:, mm + 1] - data[:, mm]

                N = np.size(np.where(~np.isnan(data_ok - rep)))
                if N == 0:
                    J[rr, cc, mm] = np.nan
                # N=length(find(~isnan(data_ok-rep)));
                else:
                    J[rr, cc, mm] = np.nansum(np.nansum(np.abs(data_ok - rep) ** 2)) / N

# sum over modes
J_ok = np.sum(J, axis=2)

v = np.nanmin(J_ok[:, :])
rr_m, cc_m = np.where(J_ok == v)

r_est = r_[rr_m]
c2_est = c2_[cc_m]

print('\n' * 5)
print('Localization done!')
print('')
print('')
print('Estimated range: ' + str(r_est[0]) + ' m')
print('Estimated seabed sound speed: ' + str(c2_est[0]) + ' m/s')
print('True values are: range ' + str(r_true[0, 0]) + ' m ; speed ' + str(c2_true[0, 0]) + ' m/s')
input('Press ENTER to plot results')


# 5.1 Plot results

print('\n' * 20)
print('The top panel of the figure shows the data (in black) and the best replicas')
print('Remember that these are not dispersion curve, but dispersion curve differences')
print('If there is not a good match between the 2, then the localization is likely wrong')
print(' ')
print('The bottom panel of the figure shows least square fit for range')
print('If the curve is not relatively smooth with a marked minimum, then')
print('the localization is likely wrong.')
print('Close the figure to continue and conclude')

rep_est = np.zeros((Nf, Nm_inv - 1))

for mm in np.arange(0, Nm_inv - 1):
    rep_est[:, mm] = (r_est / np.squeeze(vg[cc_m, mm + 1, :]) - r_est / np.squeeze(vg[cc_m, mm, :]))

plt.figure()
plt.subplot(211)
for mm in np.arange(0, Nm_inv - 1):
    plt.plot(data[:, mm + 1] - data[:, mm], freq_data[0, :], 'black')
    plt.plot(rep_est[:, mm], freq_data[0, :], 'or', fillstyle='none')

plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Dispersion curve difference')
plt.subplot(212)
plt.plot(r_ / 1000, np.squeeze(J_ok[:, cc_m]))
plt.grid()
plt.xlabel('Range (km)')
plt.title('Least square fit')
plt.show(block=False)
input('Pres ENTER to continue and exit the code')


#--------------------------------------------------------------------------------------
## 6. Concluding remarks

print('If the localization procedure went wrong')
print(' * verify that your input data are correct')
print(' * verify the the parameter grid you use is wide enough to include the true values')
print(' * look at the data/replica fit to understand what may be going on')
print('NB: replicas may contain points at early time that looks like outliers')
print('those are normal and predicted by the Pekeris waveguide model,')
print('they correspond to frequencies that are highly attenuated, ')
print('it is uncommon to see them on real data')
print(' ')
print('END')
