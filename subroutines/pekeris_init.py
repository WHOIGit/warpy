#### Warping tutorial
#### Eva Chamorro - Daniel Zitterbart - Julien Bonnel
#### May 2020


## PEKERIS init FUNCTIONS

#This scrip contains the following subroutines
#pek_init
#pek_green 



import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from pekeris import *

######################Function pek_init ############################


def pek_init(rho1, rho2, c1, c2, D, freq):
    
    Nf=len(freq)
    
    # Looking at the highest frequency for the max number of modes
    f=np.max(freq)
    toto=pek_kr_f(c1,c2,rho1,rho2,D,f)
    Nm=len(toto)
    
    # Init variables
    kr=np.zeros((Nf,Nm))
    kz=np.zeros((Nf,Nm))
    kzb=np.zeros((Nf,Nm))
    A2=np.zeros((Nf,Nm))
    
    # Loop over frequencies
    for ff in (np.arange(0,Nf)):
    
        toto=pek_kr_f(c1,c2,rho1,rho2,D,freq[ff])
        NN=len(toto)
        w=2*np.pi*freq[ff]
        
        
        # Horizontal wavenumber
        kr[ff,0:NN]=toto
    
        # Vertical wavenumber in water
        kz[ff,0:NN]=np.sqrt(w**2/c1**2-toto**2)
    
        # Vertical wavenumber in seabed
        kzb[ff,0:NN]=np.sqrt(toto**2-w**2/c2**2)
        
        # Normalization for the modal depth function
        A2[ff,0:NN]=2*rho1*kz[ff,0:NN]*kzb[ff,0:NN]/(kz[ff,0:NN]*kzb[ff,0:NN]*D - 0.5*kzb[ff,0:NN]*np.sin(2*kz[ff,0:NN]*D) + rho1/rho2*kz[ff,0:NN]*(np.sin(kz[ff,0:NN]*D)**2))
   

    # Global multiplicative factor
    i=complex(0,1)
    A2=A2*1*i*np.exp(1*i*np.pi/4)/rho1/4
    
    
    return kr, kz, A2



######################Function pek_green ############################

def pek_green(kr,kz,A2,zs,zr,r):
    
    Nz=len(zr[0])
    Nf,Nm=np.shape(kr)
    
    g=np.zeros((Nf,Nz),'complex')
    
    for ff in (np.arange(0,Nf)):
        toto=np.where(kr[ff,:] !=0)
        i=complex(0,1)
                         
        a=A2[ff,toto]*np.sin(kz[ff,toto]*zs)*np.exp(-1*i*kr[ff,toto]*r)/np.sqrt(kr[ff,toto]*r)
        b=np.sin(np.transpose(kz[ff,toto])*zr)
        g[ff,:]=np.dot(a,b)
        
    return g






   