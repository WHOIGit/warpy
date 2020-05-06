
#TIME FREQUENCY ANALYSIS FUNCTIONS

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from numpy.linalg import inv

#This scrip contains the following functions 
#fmodany
#fmlin
#fmpar
#momftfr
#tfrstft

######################Function tfrstft ############################

def tfrstft (x=np.array([[1]]),t=np.array([[False]]),N=False,h=np.array([[False]])):
    '''
    TFRSTFT Short time Fourier transform.
    [TFR,T,F]=TFRSTFT(X,T,N,H) computes the short-time Fourier
    transform of a discrete-time signal X. 

    X     : signal.
    T     : time instant(s)          (default : 1:length(X)).
    N     : number of frequency bins (default : length(X)).
    H     : frequency smoothing window, H being normalized so as to
            be  of unit energy.      (default : Hamming(N/4)).

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
    
    (xrow,xcol)=np.shape(x)
    
    if x[0,0] == 1:
        raise ValueError('At least 1 parameter is required')
        
    elif t.all() == False or N == False:
        (xrow,xcol)=np.shape(x)
        N=xrow 

    if ((xcol != 1)):
        x = np.transpose(x)  
        (xrow, xcol) = np.shape(x)
        
    
    hlength = np.floor(N / 4)
    hlength = hlength + 1 - np.remainder(hlength, 2)
    
    
    if t.all() == False:
        t=np.arange(1,xrow+1) 
        t = t[np.newaxis,:]

    if h.all() == False:
        h = np.hamming(hlength) 
        h = h[:, np.newaxis]
    
    
    (trow,tcol)=np.shape(t)
        
    if ((N < 0)): 
        raise ValueError('N must be greater than zero') 
        
    if ((xcol != 1)):
        raise ValueError('X must have one column')

    if ((trow != 1)):
        raise ValueError('T must only have one row')

    elif ((2**np.ceil(np.log2(abs(N))) != N)):
        print('For a faster computation, N should be a power of two\n')
    
    
    (hrow, hcol) = np.shape(h)
    Lh = (hrow - 1) / 2
    
    if ((hcol != 1) or (np.remainder(hrow,2) == 0)):
        raise ValueError('H must be a smoothing window with odd length')

    if (N < len(h)):
        raise ValueError('N must be greater than the window length')    
        
    
    h = h / np.linalg.norm(h)

    tfr = np.zeros((N, tcol))
          
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


################################################################


def momftfr(tfr,tmin,tmax,time):

    tfrrow, tfrcol = np.shape(tfr)
    
    '''
        
    MOMFTFR Frequency moments of a time-frequency representation.
    [TM,D2]=MOMFTFR(TFR,TMIN,TMAX,TIME) computes the frequeny 
    moments of a time-frequency representation.

    TFR    : time-frequency representation ([Nrow,Ncol]size(TFR)). 
    TMIN   : smallest column element of TFR taken into account
                                (default : 1) 
    TMAX   : highest column element of TFR taken into account
                                (default : Ncol)
    TIME   : true time instants (default : 1:Ncol)
    TM     : averaged time          (first order moment)
    D2     : squared time duration  (second order moment)

    Example :
         sig=fmlin(128,0.1,0.4); 
         [tfr,t,f]=tfrwv(sig); [tm,D2]=momftfr(tfr); 
         subplot(211); plot(f,tm); subplot(212); plot(f,D2);

    See also MOMTTFR, MARGTFR.

    F. Auger, August 1995.
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

        '''
        

    
    E  = np.sum(np.transpose(tfr[:,tmin:tmax]), axis=0)
    E=E[np.newaxis,:]
    tm = np.transpose((np.dot(time, np.transpose(tfr[:,tmin:tmax])))/E) 
    D2 = np.transpose(np.dot(time**2, np.transpose(tfr[:,tmin:tmax]))/E) - tm**2
    
    return tm, D2



################################################################


def fmpar (N=1,p1=1,p2=1,p3=1):
    
    '''
    
    FMPAR	Parabolic frequency modulated signal.
    [X,IFLAW]=FMPAR(N,P1,P2,P3) generates a signal with
    parabolic frequency modulation law.
    X(T) = exp(j*2*pi(A0.T + A1/2.T^2 +A2/3.T^3)) 

    N  : the number of points in time
    P1 : if NARGIN=2, P1 is a vector containing the three 
        coefficients [A0 A1 A2] of the polynomial instantaneous phase.
       If NARGIN=4, P1 (as P2 and P3) is a time-frequency point of 
       the form [Ti Fi].
        The coefficients (A0,A1,A2) are then deduced such that  
        the frequency modulation law fits these three points.
    P2,P3 : same as P1 if NARGIN=4.       (optional)
    X     : time row vector containing the modulated signal samples 
    IFLAW : instantaneous frequency law

    Examples :   
     [X,IFLAW]=fmpar(128,[1 0.4],[64 0.05],[128 0.4]);
     subplot(211);plot(real(X));subplot(212);plot(IFLAW);
     [X,IFLAW]=fmpar(128,[0.4 -0.0112 8.6806e-05]);
     subplot(211);plot(real(X));subplot(212);plot(IFLAW);

    See also FMCONST, FMHYP, FMLIN, FMSIN, FMODANY, FMPOWER.

    P. Goncalves - October 1995, O. Lemoine - November 1995
    Copyright (c) 1995 Rice University

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
     
        '''
    
    if N == 1:
        raise ValueError ( 'The number of parameters must be at least 2.' )
        
    if N <= 0:
        raise ValueError ('The signal length N must be strictly positive' )
        
    if p2.all == 1 :
        
        if len(p1) != 3:
            raise ValueError('Bad number of coefficients for P1')
            
            
        a0 = p1(1) 
        a1 = p1(2) 
        a2 = p1(3) 
    
    else :
        
        if (len(p1) != 2) or (len(p2) != 2) or (len(p3) != 2):
            raise ValueError('Bad number of coefficients for P1, P2, P3')
  
        if p1[0]>N or p1[0]<1:
            raise ValueError ('P1(1) must be between 1 and N')
            
        if p2[0]>N or p2[0]<1:
            raise ValueError ('P2(1) must be between 1 and N')
            
        if p3[0]>N or p3[0]<1:
            raise ValueError ('P3(1) must be between 1 and N')
            
        if p1[1]<0:
            raise ValueError ('P1(2) must be > 0')
            
        if p2[1]<0:
            raise ValueError ('P2(2) must be > 0')
            
        if p3[1]<0:
            raise ValueError ('P3(2) must be > 0')
                

        Y = np.array([p1[1],p2[1],p3[1]])
        X = np.array([[1,1,1],[p1[0], p2[0], p3[0]],[p1[0]**2, p2[0]**2, p3[0]**2]])
        coef = np.dot(Y,inv(X)) 
        a0 = coef[0] 
        a1 = coef[1] 
        a2 = coef[2] 
 

    t=np.arange(1,N+1)

    phi = 2*np.pi*(a0*t + a1/2*t**2 + a2/3*t**3) 
    iflaw =np.transpose (a0 + a1*t + a2*t**2)

    aliasing = np.where((iflaw < 0) | (iflaw > 0.5 )) 
    if len(aliasing[0]) != 0:
        print(['!!! WARNING: signal is undersampled or has negative frequencies']) 
        
    c=complex(0,1)
    x = np.transpose(np.exp(c*phi))

    
    return x, iflaw


################################################################

    
def fmlin(N=1,fnormi=1,fnormf=1,t0=1):
   

    '''
    FMLIN	Signal with linear frequency modulation.
    [Y,IFLAW]=FMLIN(N,FNORMI,FNORMF,T0) generates a linear frequency  
    modulation.
    The phase of this modulation is such that Y(T0)=1.

    N       : number of points
    FNORMI  : initial normalized frequency (default: 0.0)
    FNORMF  : final   normalized frequency (default: 0.5)
    T0      : time reference for the phase (default: N/2).
    Y       : signal
    IFLAW   : instantaneous frequency law  (optional).

    Example : 
     z=amgauss(128,50,40).*fmlin(128,0.05,0.3,50); 
     plot(real(z));

    see also FMCONST, FMSIN, FMODANY, FMHYP, FMPAR, FMPOWER.

    F. Auger, July 1995.
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
 
        '''

    if N == 1:
        raise ValueError ( 'The number of parameters must be at least 1.' )
        
    if fnormi == 1:
        fnormi=0.0
        fnormf=0.5
        t0= round(N/2)
        
    if fnormf == 1:
        fnormf=0.5
        t0 = round(N/2)
        
    if t0 == 1:
        t0 = round(N/2)


    if N <= 0:
        raise ValueError ('The signal length N must be strictly positive' )
        
    if (np.abs(fnormi) > 0.5) | (np.abs(fnormf) > 0.5):
         raise ValueError ( 'fnormi and fnormf must be between -0.5 and 0.5' ) 
    
    else:
        y=np.arange(1,N+1)
        y=y[:,np.newaxis]
        y = fnormi*(y-t0) + ((fnormf-fnormi)/(2.0*(N-1))) * ((y-1)**2 - (t0-1)**2)
        j=complex(0,1)
        y = np.exp(j*2.0*np.pi*y) 
        y=y/y[t0-1]
     
    
    iflaw=np.transpose(np.linspace(fnormi,fnormf,N))
    iflaw=iflaw[:,np.newaxis]
    
    return y,iflaw

#################################################################################


def fmodany(iflaw,t0=1):

    '''
    FMODANY Signal with arbitrary frequency modulation.
    [Y,IFLAW]=FMODANY(IFLAW,T0) generates a frequency modulated
    signal whose instantaneous frequency law is approximately given by
    the vector IFLAW (the integral is approximated by CUMSUM).
    The phase of this modulation is such that y(t0)=1.

    IFLAW : vector of the instantaneous frequency law samples.
    T0    : time reference		(default: 1).
    Y     : output signal

    Example:
     [y1,ifl1]=fmlin(100); [y2,ifl2]=fmsin(100);
     iflaw=[ifl1;ifl2]; sig=fmodany(iflaw); 
     subplot(211); plot(real(sig))
     subplot(212); plot(iflaw); 

    See also FMCONST, FMLIN, FMSIN, FMPAR, FMHYP, FMPOWER.

    F. Auger, August 1995.
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
    
        '''

    ifrow,ifcol=np.shape(iflaw)
    
    if (ifcol!=1):
         raise ValueError ('IFLAW must have one column')
            
    if (np.max(np.abs(iflaw))>0.5): 
         raise ValueError('The elements of IFLAW should not be higher than 0.5')

        
    if ((t0==0)|(t0>ifrow)):
        raise ValueError('T0 should be between 1 and len(IFLAW)')

    j=complex(0,1)
    y=np.exp(j*2.0*np.pi*np.cumsum(iflaw))
    y=y*np.conj(y[t0-1])
    
    
    return y,iflaw



    