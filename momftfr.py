import numpy as np

def momftfr(tfr, tmin, tmax, time):
    tfrrow, tfrcol = np.shape(tfr)

    E = np.sum(np.transpose(tfr[:, 0:150]), axis=0)
    E = E[np.newaxis, :]
    tm = np.transpose((np.dot(time, np.transpose(tfr[:, 0:150]))) / E)
    D2 = np.transpose(np.dot(time ** 2, np.transpose(tfr[:, tmin:tmax])) / E) - tm ** 2

    return tm, D2

