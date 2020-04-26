######################Function tfrstft ############################
import numpy as np
from scipy.fftpack import fft

def tfrstft(x, t, N, N_window):
    h = np.hamming(N_window)
    h = h[:, np.newaxis]

    (xrow, xcol) = np.shape(x)

    ## Lines added by Bonnel
    if ((xcol != 1)):
        x = np.transpose(x);  ###EvaComment:transpuesta .'
        (xrow, xcol) = np.shape(x);
    ## end of Bonnel's addition

    hlength = np.floor(N / 4);
    hlength = hlength + 1 - np.remainder(hlength, 2);

    trace = 0;

    (trow, tcol) = np.shape(t);

    (hrow, hcol) = np.shape(h);
    Lh = (hrow - 1) / 2;

    h = h / np.linalg.norm(h);

    tfr = np.zeros((N, tcol));

    for icol in range(tcol):
        ti = t[0, icol];
        tau = np.arange(-np.min([np.round(N / 2) - 1, Lh, ti - 1]), np.min([np.round(N / 2) - 1, Lh, xrow - ti]) + 1);
        indices = np.array(np.remainder(N + tau, N) + 1, dtype='int')
        a = np.array(Lh + 1 + tau, dtype='int')
        b = np.array(ti + tau, dtype='int')
        c = x[b - 1, :] * np.conj(h[a - 1])
        tfr[indices - 1, icol] = c[:, 0]

    tfr = fft(tfr, axis=0)

    return tfr