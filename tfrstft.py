import numpy as np
from scipy.fftpack import fft


def tfrstft(x=np.array([[1]]), t=np.array([[False]]), N=False, N_window=False, trace=0):
    '''
    TFRSTFT Short time Fourier transform.
    [TFR,T,F]=TFRSTFT(X,T,N,H,TRACE) computes the short-time Fourier 
    transform of a discrete-time signal X. 

    X     : signal.
    T     : time instant(s)          (default : 1:length(X)).
    N     : number of frequency bins (default : length(X)).
    H     : frequency smoothing window, H being normalized so as to
            be  of unit energy.      (default : Hamming(N/4)). 
    TRACE : if nonzero, the progression of the algorithm is shown
                                     (default : 0).
    TFR   : time-frequency decomposition (complex values). The
           frequency axis is graduated from -0.5 to 0.5.
    F     : vector of normalized frequencies.

    Example :
    sig=[fmconst(128,0.2);fmconst(128,0.4)]; tfr=tfrstft(sig);
    subplot(211); plt.pcolormesh(abs(tfr));
    subplot(212); plt.pcolormesh(angle(tfr));

    See also all the time-frequency representations listed in
    the file CONTENTS (TFR*)

    F. Auger, May-August 1994, July 1995.
    Copyright (c) 1996 by CNRS (France).

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA


     This matlab function has been extracted from the Time Frequency Toolbox
     (TFTB) by Julien Bonnel (Woods Hole Oceanographic Institution) on August
     31, 2019. The original toolbox can be downloaded here:
     http://tftb.nongnu.org/ (last seen on August 31, 2019).
     The matlab function has been slightly modified by Julien Bonnel so that
     it can be used without the full toolbox. Each modification is clearly
     identified using comments.

    '''

    (xrow, xcol) = np.shape(x)

    if x[0,0] == 1:
        raise ValueError('At least 1 parameter is required')

    elif t.all() == False or N == False:
        (xrow, xcol) = np.shape(x)
        N = xrow

    ## Lines added by Bonnel

    #if xrow >= 2:
        #raise ValueError('X must be a vector')  # X must only have one row???

    if ((xcol != 1)):
        x = np.transpose(x)
        (xrow, xcol) = np.shape(x)

    ## end of Bonnel's addition

    hlength = np.floor(N / 4)
    hlength = hlength + 1 - np.remainder(hlength, 2)

    if t.all() == False:
        # t=1:xrow; h = tftb_window(hlength); trace=0; ### line commented by Bonnel
        t = np.arange(1, xrow + 1)  ### line added by Bonnel
        t = t[np.newaxis, :]

    if N_window == False:
        # h = tftb_window(hlength); trace = 0;  ### line commented by Bonnel
        h = np.hamming(hlength)  ### line added by Bonnel
        h = h[:, np.newaxis]


    elif N_window != False:
        h = np.hamming(N_window)
        h = h[:, np.newaxis]

    (trow, tcol) = np.shape(t)

    if ((N < 0)):  ### line added by Bonnel
        raise ValueError('N must be greater than zero')  ### line added by Bonnel

    if ((xcol != 1)):
        raise ValueError('X must have one column')

    if ((trow != 1)):
        raise ValueError('T must only have one row')

    elif ((2 ** np.ceil(np.log2(abs(N))) != N)):
        print('For a faster computation, N should be a power of two\n')

    (hrow, hcol) = np.shape(h)
    Lh = (hrow - 1) / 2

    if ((hcol != 1) or (np.remainder(hrow, 2) == 0)):
        raise ValueError('H must be a smoothing window with odd length')

    if (N < len(h)):
        raise ValueError('N must be greater than the window length')

    h = h / np.linalg.norm(h)

    tfr = np.zeros((N, tcol))

    if trace:
        print('Short-time Fourier transform')

    for icol in range(tcol):
        ti = t[0, icol];
        tau = np.arange(-np.min([np.round(N / 2) - 1, Lh, ti - 1]), np.min([np.round(N / 2) - 1, Lh, xrow - ti]) + 1);
        indices = np.array(np.remainder(N + tau, N) + 1, dtype='int')
        a = np.array(Lh + 1 + tau, dtype='int')
        b = np.array(ti + tau, dtype='int')
        c = x[b - 1, :] * np.conj(h[a - 1])
        tfr[indices - 1, icol] = c[:, 0]

    tfr = fft(tfr, axis=0)

    # if trace:
    # fprintf('\n')

    if np.remainder(N, 2) == 0:
        f1 = np.arange(0, N / 2, 1)
        f2 = -np.arange(1, (N / 2) + 1, 1)
        f3 = f2[::-1]
        f = np.concatenate((f1, f3))
        f = f[:, np.newaxis]
        f = f / N

    else:
        f1 = np.arange(0, (N - 1) / 2, 1)
        f2 = -np.arange(1.5, ((N - 1) / 2) + 1, 1)
        f3 = f2[::-1]
        f = np.concatenate((f1, f3))
        f = f4[:, np.newaxis]
        f = f5 / N

    return tfr, t, f