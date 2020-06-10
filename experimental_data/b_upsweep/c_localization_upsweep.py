import os
#os.chdir("/Users/evachamorro/Desktop/stage_M2/biblio/basic_stuff/warping_tuto/supplementary_material/python code/functions")
#**Put here the directory where you have the file with your function**
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from datetime import date
from pekeris import *
from hamilton import *
from datetime import date

#os.chdir('/Users/evachamorro/Desktop/stage_M2/biblio/basic_stuff/warping_tuto/supplementary_material/python code/experimental_data/b_fm_upsweep')
##**Put here the directory where you were working**

## 2. Parameters for localization

#### First, all the parameters that we think we know
D = 55  ### water depth
# c1=1442 ### sound speed in wat
c1 = 1500
rho1 = 1  ### density in water

#### now the search grids for the parameters to be estimated
r_ = np.arange(10000, 20100, 100)
c2_ = np.arange(1550, 1810, 10)
### NB: dt values should be roughly between rmin/c1 and rmax/c1. One can start with wide
###     search bounds and coarse steps for dt, and gradually narrow the bounds and decrease the steps.

## 3. Load data

print('Select the .mat file with the dispersion curves you want to use for localization')
print('(it has been created by b_filtering.m)')
print('NB: this code is only to localize impulsive sources. Do not use it for sources that are not perfect impulses')

today = date.today()
dat = sio.loadmat(os.getcwd()+'/upsweep_modes_20200609T155921'+ '.mat')
#dat = sio.loadmat(os.getcwd()+ '/upsweep_modes_' + str(today)+ '.mat')

data=dat['data']
freq_data=dat['freq_data']
Nmode=dat['Nmode']

input('PresS ENTER to continue')
print('\n' * 20)
print('')
print('These are estimated dispersion curves obtained with')
print('the previous code (b_filtering.m)')
print('They will be used as data for localization')


plt.figure()
plt.plot(data[:,0], freq_data[0,:],'black')
plt.plot(data[:,1], freq_data[0,:],'black')
plt.plot(data[:,2], freq_data[0,:],'black')
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.grid()
plt.title('Dispersion curves')
plt.show(block=True)

## 4. Compute replicas

Nf=len(freq_data[0,:])
Nr=len(r_)
Nc=len(c2_)
Nm=int(Nmode)

vg=np.zeros((Nc,Nm, Nf))
m_min=1
m_max=Nm

fmin=freq_data[0,0]
fmax=freq_data[0,-1]
df=freq_data[0,1]-freq_data[0,0]

print('\n' * 20)
print('The first step is to compute replicas')
print('Computing replicas ....')

for cc in (np.arange(0, Nc)):
    c2 = c2_[cc]
    rho2 = hamilton(c2)
    [vg[cc, :, :], f_rep] = pek_vg(fmin, fmax, m_min, m_max, c1, c2, rho1, rho2, D, df)

print('Replicas computed!')
print('Continue')

## 5. Localization

##### we need common frequencies between modes. Let us verify the data
flag_data = 0
for mm in np.arange(0, Nm - 1):
    data_ok = data[:, mm + 1] - data[:, mm]
    data_ok_ok = data_ok[~np.isnan(data_ok)]
    if len(data_ok_ok) != 0:
        flag_data = 1

if flag_data == 0:  ### there are no common frequency band between modes
    print('\n' * 20)
    print('There are no common frequency band between modes')
    print('It is impossible to use Eq. (29)')
    print('Go back to the previous code, and redo the dispersion curve estimation')
    print('Make sure to have an overlap in frequency between the modes')
    print(' ')
    print('END')

else:  ## there are common frequencies and we can proceed
    J = np.zeros((Nr, Nc, Nm - 1))
    print('\n' * 20)
    print('Starting localization ...')

    for rr in (np.arange(0, Nr)):
        # waitbar(step/Nsteps,www,'Localization in progress ...');
        r = r_[rr]
        for cc in (np.arange(0, Nc)):
            for mm in (np.arange(0, Nm - 1)):
                rep = (r / np.squeeze(vg[cc, mm + 1, :]) - r / np.squeeze(vg[cc, mm, :]))
                data_ok = data[:, mm + 1] - data[:, mm]
                d = data_ok - rep
                data_ok_nan = data_ok[~np.isnan(d)]
                N = len(data_ok_nan)
                if N == 0:
                    J[rr, cc, mm] = np.nan
                else:
                    J[rr, cc, mm] = np.nansum(np.nansum(np.abs(data_ok - rep) ** 2)) / N

    J_ok = np.sum(J, axis=2)

    v = np.nanmin(J_ok[:, :])
    rr_m, cc_m = np.where(J_ok == v)

    r_est = r_[rr_m]
    c2_est = c2_[cc_m]


    print('\n' * 5)
    print('Localization done!')
    print(['Estimated range: ' + str(r_est) + ' m'])
    print(['Estimated seabed sound speed: ' + str(c2_est) + ' m/s'])
    print('Continue to plot results')



## 6. Plot results

rep_est = np.zeros((Nf, Nm - 1))
for mm in np.arange(0, Nm - 1):
    rep_est[:, mm] = (r_est / np.squeeze(vg[cc_m, mm + 1, :]) - r_est / np.squeeze(vg[cc_m, mm, :]))

print('\n' * 20)
print('The top panel of the figure shows the data (in black) and the best replicas')
print('Remember that these are not dispersion curve, but dispersion curve differences')
print('If there is not a good match between the 2, then the localization is likely wrong')
print(' ')
print('The bottom panel of the figure shows least square fit for range')
print('If the curve is not relatively smooth with a marked minimum, then')
print('the localization is likely wrong.')
print('Press any key to continue')
print(' ')
print('If are working on the upsweep provided in the tutorial, our range estimate')
print('is around 16.5 km. What did you find?')

plt.figure()

plt.subplot(211)
for mm in range(Nm - 1):
    plot = data[:, mm + 1] - data[:, mm]
    plt.plot(plot, freq_data[0, :], 'black')
    plt.plot(rep_est[:, mm], freq_data[0, :], 'or', fillstyle='none')


plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Dispersion curve difference')

plt.subplot(212)
plt.plot(r_ / 1000, np.squeeze(J_ok[:, cc_m]))
plt.grid()
plt.xlabel('Range (km)')
plt.title('Least square fit')

plt.show(block=True)


print(' ')
print('END')
