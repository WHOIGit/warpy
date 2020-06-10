## Warping tutorial
#### c_localization, deconv

#### THIS CODE IS TO LOCALIZE IMPULSIVE SOURCE ONLY

##### May 2020
###### Eva Chamorro - Daniel Zitterbart - Julien Bonnel

## 1. Import packages

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

#os.chdir('/Users/evachamorro/Desktop/stage_M2/biblio/basic_stuff/warping_tuto/supplementary_material/python code/experimental_data/c_css_deconv')
##**Put here the directory where you were working**

## 2. Parameters for localization

#### First, all the parameters that we think we know
D = 69.5  ### water depth
c1 = 1464.5  ### sound speed in wat
rho1 = 1  ### density in water

#### now the search grids for the parameters to be estimated
r_ = np.arange(3000, 6100, 100)
c2_ = np.arange(1470, 2020, 20)
dt_ = np.arange(-4, -1.99,
                0.01)  ## since we work with impulsive source, we will use Eq. (28) and need to estimate a dt variable
### NB: dt values should be roughly between rmin/c1 and rmax/c1. One can start with wide
###     search bounds and coarse steps for dt, and gradually narrow the bounds and decrease the steps.


## 3. Load data

print('Select the .mat file with the dispersion curves you want to use for localization')
print('(it has been created by b_filtering.m)')
print('NB: this code is only to localize impulsive sources.')
print('If the source is not impulsive make sure to do source deconvolution first')

today = date.today()
dat = sio.loadmat(os.getcwd()+'/css_ready_to_warp_modes_' + str(today)+ '.mat')


data=dat['data']
freq_data=dat['freq_data']
Nmode=dat['Nmode']

print('')

print('These are estimated dispersion curves obtained with')
print('the previous code (b_filtering.m)')
print('They will be used as data for localization')
print('')
print('Close the figure to continue')

plt.figure()
for i in range (int(Nmode)):
    plt.plot(data[:,i], freq_data[0,:],'black')

plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.grid()
plt.title('Dispersion curves')
plt.show(block=True)

## 4. Compute replicas

Nf=len(freq_data[0,:])
Nr=len(r_)
Nc=len(c2_)
Nt=len(dt_)
Nm=int(Nmode)

vg=np.zeros((Nc,Nm, Nf))
m_min=1
m_max=Nm

fmin=freq_data[0,0]
fmax=freq_data[0,-1]
df=freq_data[0,1]-freq_data[0,0]

print('The first step is to compute replicas')
print('Computing replicas ....')

for cc in (np.arange(0, Nc)):
    c2 = c2_[cc]
    rho2 = hamilton(c2)
    [vg[cc, :, :], f_rep] = pek_vg(fmin, fmax, m_min, m_max, c1, c2, rho1, rho2, D, df)

print('Replicas computed!')



## 5. Localization

print('Starting localization ...')
print('Since the source is impulsive we use Eq. (19)')

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
print([
          'If one of the estimated parameter is stuck to a boundary of its search grid, the localization result is likely wrong'])
print('Continue to plot results')


## 6. Plot results

rep_est=r_est/np.squeeze(vg[cc_m,:,:])+dt_est

print('The top panel of the figure shows the data (in black) and the predicted dispersion curves')
print('(i.e. the best replicas) in black. If there is not a good match')
print('between the 2, then the localization is likely wrong')
print(' ')
print('The bottom panel of the figure shows least square fit for range')
print('If the curve is not relatively smooth with a marked minimum, then')
print('the localization is likely wrong.')
print(' ')
print('If are working on the gunshot provided in the tutorial, our range estimate')
print('is around 4.5 km. What did you find?')
print(' ')
print(' ')

plt.figure()
plt.subplot(211)
for i in range (int(Nmode)):

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

plt.show(block=True)

print(' ')
print('END')

