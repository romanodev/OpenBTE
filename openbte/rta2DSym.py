import deepdish as dd
import sys
import numpy as np
import scipy
import deepdish as dd
from .utils import *



def generate_rta2DSym(**argv):

 #Import data----
 data = dd.io.load('rta.h5')
 #Get options----
 n_phi = int(argv.setdefault('n_phi',48))
 n_mfp = int(argv.setdefault('n_mfp',50))
 n_theta = 24

 nm = n_phi * n_mfp

 #Create sampled MFPs
 #Polar Angle---------
 Dphi = 2*np.pi/n_phi
 phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
 polar_ave = np.array([np.sin(phi),np.cos(phi),np.zeros(n_phi)]).T


 #Import data-----------
 data = dd.io.load('rta.h5')
 f = np.divide(np.ones_like(data['tau']), data['tau'], out=np.zeros_like(data['tau']), where=data['tau']!=0)
 tc = data['C']*f
 tc /= tc.sum()
 Jc = np.einsum('k,ki->ki',data['C'],data['v'][:,:2])
 nm = len(tc)
 mfp_bulk = np.einsum('ki,k->ki',data['v'][:,:2],data['tau'])
 r = np.array([np.linalg.norm(m) for m in mfp_bulk])
 phi_bulk = np.array([np.arctan2(m[0],m[1]) for m in mfp_bulk])
 phi_bulk[np.where(phi_bulk < 0) ] = 2*np.pi + phi_bulk[np.where(phi_bulk <0)]
 kappa = data['kappa']

 mfp_sampled = np.logspace(-10,np.log10(max(r)*1.01),n_mfp)#min MFP = 1e-1 nm

 #-----------------------
 n_mfp_bulk = len(mfp_bulk) 
 n_mfp = argv.setdefault('n_mfp',100)
 mfp = np.logspace(-1,np.log10(max(r)*1.01),n_mfp)#min MFP = 1e-2 

 n_mfp = len(mfp)
 temp_coeff = np.zeros((n_mfp,n_phi))
 kappa_directional = np.zeros((n_mfp,n_phi,2)) 

 lo = 0
 for m in range(nm):

    if r[m] > 0:  
     (m1,a1,m2,a2) = interpolate(mfp_sampled,r[m],bounds='extent')
     (p1,b1,p2,b2) = interpolate(phi,phi_bulk[m],bounds='periodic',period=2*np.pi)

     temp_coeff[m1,p1] += tc[m]*a1*b1
     temp_coeff[m1,p2] += tc[m]*a1*b2
     temp_coeff[m2,p1] += tc[m]*a2*b1
     temp_coeff[m2,p2] += tc[m]*a2*b2
    
     kappa_directional[m1,p1] += Jc[m]*a1*b1
     kappa_directional[m1,p2] += Jc[m]*a1*b2
     kappa_directional[m2,p1] += Jc[m]*a2*b1
     kappa_directional[m2,p2] += Jc[m]*a2*b2

    else: 
     lo += tc[m]   

 temp_coeff *=(1+lo)

 rhs_average = mfp_sampled*mfp_sampled/2

 return {'tc':temp_coeff,\
         'sigma':kappa_directional,\
         'kappa':kappa,\
         'mfp_average':rhs_average*1e18,\
         'VMFP':polar_ave,\
         'mfp_sampled':mfp_sampled,\
         'model':np.array([8])}


