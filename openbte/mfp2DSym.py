import deepdish as dd
import sys
import numpy as np
import scipy
import deepdish as dd
from .utils import *



def generate_mfp2DSym(**argv):

 #Import data----
 data = dd.io.load('mfp.h5')
 kappa_bulk = data['K']
 mfp_bulk = data['mfp']

 kappa= np.eye(3) * np.sum(kappa_bulk)
 #Get options----
 n_phi = int(argv.setdefault('n_phi',48))
 n_mfp = int(argv.setdefault('n_mfp',50))
 n_theta = 24

 nm = n_phi * n_mfp

 #Create sampled MFPs
 n_mfp_bulk = len(mfp_bulk)
 mfp_sampled = np.logspace(min([-2,np.log10(min(mfp_bulk)*0.99)]),np.log10(max(mfp_bulk)*1.01),n_mfp)#min MFP = 1e-2 nm 


 #Polar Angle---------
 Dphi = 2*np.pi/n_phi
 phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
 #--------------------

 #Get coeff matrix---
 #coeff = np.zeros((n_mfp,n_mfp_bulk))
 #for m,rr in enumerate(mfp_bulk):
 #   (m1,a1,m2,a2) = get_linear_indexes(mfp_sampled,rr,scale='linear',extent=True)
 #   coeff[m1,m] = a1
 #   coeff[m2,m] = a2
 #-----------------------

 #---------------------
   
 #Azimuthal Angle------------------------------
 Dtheta = np.pi/n_theta/2.0
 theta = np.linspace(Dtheta/2.0,np.pi/2.0 - Dtheta/2.0,n_theta)
 dtheta = 2.0*np.sin(Dtheta/2.0)*np.sin(theta)   
 domega = np.outer(dtheta,Dphi * np.ones(n_phi))

 #Compute directions---
 polar = np.array([np.sin(phi),np.cos(phi),np.zeros(n_phi)]).T
 azimuthal = np.array([np.sin(theta),np.sin(theta),np.zeros(n_theta)]).T
 direction = np.einsum('ij,kj->ikj',azimuthal,polar)
 
 #Compute average---
 ftheta = (1-np.cos(2*theta)*np.sinc(Dtheta/np.pi))/(np.sinc(Dtheta/2/np.pi)*(1-np.cos(2*theta)))
 fphi= np.sinc(Dphi/2.0/np.pi)
 polar_ave = polar*fphi
 azimuthal_ave = np.array([ftheta*np.sin(theta),ftheta*np.sin(theta),np.zeros(n_theta)]).T   
 direction_ave = np.einsum('ij,kj->ikj',azimuthal_ave,polar_ave)
 direction_int = np.einsum('ijl,ij->ijl',direction_ave,domega)
 #------------------------------------------------------

 n_mfp_bulk = len(mfp_bulk) 
 n_mfp = argv.setdefault('n_mfp',100)
 mfp = np.logspace(min([-2,np.log10(min(mfp_bulk)*0.99)]),np.log10(max(mfp_bulk)*1.01),n_mfp)#min MFP = 1e-2 

 n_mfp = len(mfp)
 temp_coeff = np.zeros((n_mfp,n_phi))
 kappa_directional = np.zeros((n_mfp,n_phi,3)) 
 suppression = np.zeros((n_mfp,n_phi,n_mfp_bulk,3)) 
  
 for p in range(n_phi): 
  for t in range(n_theta): 
    for m in range(n_mfp_bulk):
      reduced_mfp = mfp_bulk[m]*ftheta[t]*np.sin(theta[t])
      (m1,a1,m2,a2) = get_linear_indexes(mfp,reduced_mfp,scale='linear',extent=True)
      index_1 = p*n_mfp + m1
      index_2 = p*n_mfp + m2
      if mfp_bulk[m] > 0:
       kappa_directional[m1,p]         += 3 * a1 * 2 * kappa_bulk[m]/mfp_bulk[m]/4.0/np.pi * direction_ave[t,p]*domega[t,p]
       kappa_directional[m2,p]         += 3 * a2 * 2 * kappa_bulk[m]/mfp_bulk[m]/4.0/np.pi * direction_ave[t,p]*domega[t,p]
       suppression[m1,p,m]         += 3 * a1 * 2 * 1/mfp_bulk[m]/4.0/np.pi * direction_ave[t,p]*domega[t,p]
       suppression[m2,p,m]         += 3 * a2 * 2 * 1/mfp_bulk[m]/4.0/np.pi * direction_ave[t,p]*domega[t,p]
       temp_coeff[m1,p] += a1 * kappa_bulk[m]/mfp_bulk[m]/mfp_bulk[m]*domega[t,p]
       temp_coeff[m2,p] += a2 * kappa_bulk[m]/mfp_bulk[m]/mfp_bulk[m]*domega[t,p]


 #replicate bulk values---

 tc = temp_coeff/np.sum(temp_coeff)
 Wod = np.tile(tc,(nm,1))

 angle_map = np.arange(n_phi)
 angle_map = np.repeat(angle_map,n_mfp)

 #repeat angle---
 kappa_directional[:,:,2] = 0 #Enforce zeroflux on z (for visualization purposed)
 F = np.einsum('m,pi->mpi',mfp,polar_ave)

 rhs_average = mfp_sampled*mfp_sampled/2
 
 #Final----

 return {'tc':tc,\
         'sigma':kappa_directional,\
         'kappa':kappa,\
         'scale':np.ones((n_mfp,n_phi)),\
         'mfp_average':rhs_average*1e18,\
         'VMFP':polar_ave,\
         'mfp_sampled':mfp,\
         'suppression':suppression,\
         'kappam':kappa_bulk,\
         'mfp_bulk':mfp_bulk}


