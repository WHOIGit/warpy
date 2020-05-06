
########## WARPING FUNCTIONS ##################

import numpy as np
import matplotlib.pyplot as plt

#This scrip contains the following functions:
#warp_temp_exa
#iwarp_temp_exa
#warp_t
#iwarp_t

##################### Function warp_temp_exa ##################
def warp_temp_exa(s, fs, r, c):
    '''
    warp_temp_exa.m
    Julien Bonnel-Eva Chamorro, Woods Hole Oceanographic Institution
    April 2020

    Warping main function
    Inputs
    s: original signal (time domain)
    fs: sampling frequency
    r,c: range and water sound speed (warping parameters)
    Outputs
    fs_w: sampling frequency of the warped signal

    '''


    ## Step 1: preliminary computations
    s = np.real(s)
    N = len(s[0, :])
    dt = 1 / fs

    tmin = r / c + dt
    tmax = N / fs + r / c

    ## Step 2: new time step

    dt_w = iwarp_t(tmax, r, c) - iwarp_t(tmax - dt, r, c)

    ## Step 3: new sampling frequency
    fs_w = 2 / dt_w

    ## Step 4: new number of points
    t_w_max = iwarp_t(tmax, r, c)
    M = np.ceil((t_w_max) * fs_w)

    ## Step 5: warped signal computation

    # Warped time axis, uniform sampling
    t_w = np.arange(0, M) / fs_w

    # Warped time axis, non-uniform sampling (starts from r/c)

    t_ww = warp_t(t_w, int(r), int(c))

    # factor for energy conservation
    coeff = np.sqrt(t_w / t_ww)

    # Start exact interpolation (Shannon)
    s_aux = np.tile(s, ((int(M), 1)))
    a = np.conj(t_ww).T
    aux1 = np.tile(a, (1, N))  # size=(M,1) -> repmat(1,N)
    b = tmin + np.arange(0, N) / fs
    aux2 = np.tile(b, (int(M), 1))  # size=(1,N) -> repmat(M,1)
    aux = np.sinc(fs * (aux1 - aux2))  # size=(M,N)
    # end of exact interpolation --> interpolated signal is sum(s_aux.*aux,2)

    # Final warped signal
    aux3 = np.sum(s_aux * aux, axis=1)
    aux3 = aux3[:, np.newaxis]
    s_w = np.conj(coeff).T * aux3

    return s_w, fs_w

################################################################

##################### Function iwarp_temp_exa ##################

def iwarp_temp_exa(s_w, fs_w, r, c, fs, N):
    '''

     iwarp_temp_exa.m
     Julien Bonnel, Woods Hole Oceanographic Institution
     March 2019

     Inverse warping main function
     Inputs
     s_w: warped signal (time domain)
     fs_w: sampling frequency of the warped signal
     r,c: range and water sound speed (warping parameters)
     fs: sampling frequency of the original/unwarped signals
     N: number of samples of the original/unwarped signals
     Outputs
     s_r: signal after inverse warping
    '''

    # Preliminary steps
    M = len(s_w)

    if (np.ndim(s_w) >= 2):
        s_w = np.transpose(s_w)

    ## Inverse warping computation

    # Time axis, uniform sampling (starts from r/c+dt)
    t = np.arange(1, N + 1) / fs + r / c

    # Time axis, non-uniform sampling
    t_iw = iwarp_t(t, int(r), int(c))

    # factor for energy conservation
    coeff = np.sqrt(t / t_iw)

    # Start exact interpolation (Shannon)
    s_aux = np.tile(s_w, (N, 1))  # initial signal replicated N times (N rows)
    a = np.conj(t_iw).T
    aux1 = np.tile(a, (1, M))
    aux2 = np.tile((np.arange(0, M)) / fs_w, (N, 1))
    aux = np.sinc(fs_w * (aux1 - aux2))
    # end of exact interpolation --> interpolated signal is sum(s_aux.*aux,2)

    # Final warped signal
    aux3 = np.sum(s_aux * aux, axis=1)
    aux3 = aux3[:, np.newaxis]
    s_r = np.real(np.conj(coeff).T * aux3)

    return s_r

################################################################

##################### Function warp_t ##################

def warp_t(t, r, c):
    '''
     warp_t.m
     Julien Bonnel, Woods Hole Oceanographic Institution
     March 2019

     warping subroutine

     '''
    t_w = np.sqrt(t ** 2 + r ** 2 / c ** 2)
    return t_w

################################################################

##################### Function iwarp_t ##################

def iwarp_t(t_w,r,c):
    '''
     iwarp_t.m
     Julien Bonnel, Woods Hole Oceanographic Institution
     March 2019

     warping subroutine

    '''
    t=np.sqrt(t_w**2-(r/c)**2)
    return t

################################################################