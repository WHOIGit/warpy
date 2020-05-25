#### Warping tutorial

#### Eva Chamorro - Daniel Zitterbart - Julien Bonnel
#### May 2020

## PEKERIS FUNCTIONS

#This scrip contains the following functions 
#pek_kr_f
#pek_vg
#pek_init
#pek_green 



import numpy as np
import cmath
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.optimize import fsolve



######################Function pek_kr_f############################

def pek_kr_f(c1,c2,rho1,rho2,D,freq):
    
    '''
     Inputs :
         c1 / rho1 : sound speed / density in water
         c2 / rho2 : sound speed / density in the seabed
         D : water depth
         freq : frequency
     Outputs :
         root_ok_pek : vector of wavenumbers 
         (size = number of modes at the considered frequency)
         
         '''

    jj=complex(0,1)

    # wavenumbers are between k1 and k2
    w=2*np.pi*freq
    k1=w/c1
    k2=w/c2

    # Search space for wavenumber
    n_test=5000
    kr=np.linspace(k2*(1+10**-15),k1*(1-10**-15),n_test)

    # Pekeris waveguide equation
    
    func_pek=lambda krm: np.tan(cmath.sqrt(k1**2-krm**2)*D)+rho2*cmath.sqrt(k1**2-krm**2)/rho1/(cmath.sqrt(krm**2-k2**2))


    
    # Let's work in small interval to help solving the equation 
    toto_pek=np.tan(np.sqrt(k1**2-kr**2)*D)+rho2*np.sqrt(k1**2-kr**2)/rho1/(np.sqrt(kr**2-k2**2))

    j=0

    guess_pek=np.array([])

    for i in (np.arange(0,n_test-1)): 
        
        if toto_pek[i]/toto_pek[i+1]<=0 :
            
            guess_pek = np.append(guess_pek, np.array([kr[i],kr[i+1]]))
            a=int(len(guess_pek)/2)
            guess_pek=np.reshape(guess_pek, (a,2))
            
            
    root_ok_pek=np.array([])

    # Equation is solved with fzero
    
    if len(guess_pek) !=0:
        root_pek=np.zeros([len(guess_pek)])
        
        for j in np.arange(0,len(guess_pek)):
            arr= np.array([guess_pek[j,0]])
            root_pek[j]=fsolve(func_pek,arr)
            
        for i in (np.arange(0,len(root_pek))):
    
            if np.abs(func_pek(root_pek[i]))<1 and np.isreal(root_pek[i]):
                root_ok_pek = np.append(root_ok_pek, np.array([root_pek[i]]))
            
            
    # We order the wavenumbers
    root_ok_pek=root_ok_pek[::-1]  
    
    root_ok_pek=root_ok_pek[::2]
    
    return root_ok_pek 



######################Function pek_vg ############################

def pek_vg(fmin,fmax,m_min,m_max,c1,c2,rho1,rho2,D,df):
    '''
     Inputs :
         c1 / rho1 : sound speed / density in water
         c2 / rho2 : sound speed / density in the seabed
         D : water depth
         freq : frequency
     Outputs :
         vg : matrix of group speed (size Nmode * Nfreq)
         f_ : frequency axis
        '''
    f_=np.arange(fmin,fmax+df,df)
    Nf=len(f_)
    A=rho2/rho1
    
    vg=np.zeros((m_max-m_min+1,Nf))

    ff=1

    for f in (f_):  
    
        w=2*np.pi*f
        # Horizontal wavenumbers
        kr = pek_kr_f(c1,c2,rho1,rho2,D,f)
        
  
        # No need to start computation if there is no propagating mode
        if len(kr) !=0:
            m_max_f=len(kr)
        
        
            if (m_min <= m_max_f):
                m_max_calc=min(m_max_f,m_max)
                # in a Pekeris waveguide, we have an exact formula to compute
                # group speed from  wavenumber
                ind_m=1
            
                for m in (np.arange(m_min-1,m_max_calc)):              
                    k=kr[m]
                    # vertical wavenumber in water
                    g1=np.sqrt(w**2/c1**2-k**2)
                    # vertical wavenumber in seabed
                    g2=np.sqrt(k**2-w**2/c2**2)
                          
                    # group speed computation (no need to numerically
                    # approximate dk/df)
                    a=A/(g2**2+A**2*g1**2)
                
                    # vg(ind_m,ff)=k./w*c1^2*c2^2.*(g2*D+a.*(g1.^2+g2.^2))./(c2^2*(g2*D+a.*g2.^2)+c1^2*a.*g1.^2); 
                    p=k/w*c1**2*c2**2*(g2*D+a*(g1**2+g2**2))/(c2**2*(g2*D+a*g2**2)+c1**2*a*g1**2)
                    vg[ind_m-1,ff-1]=k/w*c1**2*c2**2*(g2*D+a*(g1**2+g2**2))/(c2**2*(g2*D+a*g2**2)+c1**2*a*g1**2)
                    ind_m=ind_m+1
                
        ff=ff+1
    
    vg[vg==0]=np.nan   
   
    return vg, f_


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



