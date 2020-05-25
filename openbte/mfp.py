import deepdish as dd
import sys
import numpy as np
import scipy
import deepdish as dd
from .utils import *

def generate_mfp(**argv): 

   #Polar Angle-----
   n_phi = int(argv.setdefault('n_phi',48)); Dphi = 2.0*np.pi/n_phi
   phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
   #--------------------
   #Azimuthal Angle------------------------------
   n_theta = int( argv.setdefault('n_theta',24))
   sym = 1

   Dtheta = np.pi/n_theta
   theta = np.linspace(Dtheta/2.0,np.pi - Dtheta/2.0,n_theta)
   dtheta = 2.0*np.sin(Dtheta/2.0)*np.sin(theta)

   #Compute directions---
   polar = np.array([np.sin(phi),np.cos(phi),np.ones(n_phi)]).T
   azimuthal = np.array([np.sin(theta),np.sin(theta),np.cos(theta)]).T
   direction = np.einsum('ij,kj->ikj',azimuthal,polar)

   ftheta = (1-np.cos(2*theta)*np.sinc(Dtheta/np.pi))/(np.sinc(Dtheta/2/np.pi)*(1-np.cos(2*theta)))
   fphi= np.sinc(Dphi/2.0/np.pi)
   polar_ave = np.array([fphi*np.ones(n_phi),fphi*np.ones(n_phi),np.ones(n_phi)]).T
   azimuthal_ave = np.array([ftheta,ftheta,np.cos(Dtheta/2)*np.ones(n_theta)]).T

   direction_ave = np.multiply(np.einsum('ij,kj->ikj',azimuthal_ave,polar_ave),direction)
   domega = np.outer(dtheta, Dphi * np.ones(n_phi))


   n_mfp = argv.setdefault('n_mfp',50)
   #Total numbre of momentum discretization----
   nm = n_phi * n_mfp * n_theta
   #------------------------------------------

   #load file
   #Import data----
   data = dd.io.load('mfp.h5')
   kappa_bulk = data['K']
   mfp_bulk = data['mfp']
   kappa = data['kappa']
   #-----------------------

   n_mfp_bulk = len(mfp_bulk) 
   mfp_sampled = np.logspace(min([-2,np.log10(min(mfp_bulk)*0.99)]),np.log10(max(mfp_bulk)*1.01),n_mfp)#min MFP = 1e-2 nm 
   n_mfp = len(mfp_sampled)
   temp_coeff = np.zeros(nm) 
   kappa_directional = np.zeros((n_mfp,n_theta*n_phi,3)) 
   suppression = np.zeros((n_mfp,n_phi*n_theta,n_mfp_bulk,3)) 
   temp_coeff = np.zeros((n_mfp,n_theta*n_phi))
   kappa_m = np.zeros(n_mfp)
   for m in range(n_mfp_bulk):
    (m1,a1,m2,a2) = get_linear_indexes(mfp_sampled,mfp_bulk[m],scale='linear',extent=True)
    kappa_m[m1] += a1*kappa_bulk[m] 
    kappa_m[m2] += a2*kappa_bulk[m] 
    for t in range(n_theta): 
     for p in range(n_phi): 
      index = t * n_phi + p
      factor = kappa_bulk[m]/4.0/np.pi*domega[t,p]
      kappa_directional[m1,index] += 3 * a1 * factor/mfp_bulk[m]*direction_ave[t,p]*sym
      kappa_directional[m2,index] += 3 * a2 * factor/mfp_bulk[m]*direction_ave[t,p]*sym
      temp_coeff[m1,index] += a1 * kappa_bulk[m]/mfp_bulk[m]/mfp_bulk[m]*domega[t,p]*sym
      temp_coeff[m2,index] += a2 * kappa_bulk[m]/mfp_bulk[m]/mfp_bulk[m]*domega[t,p]*sym

      suppression[m1,index,m] += 3 * a1 /mfp_bulk[m]*direction_ave[t,p]*sym*domega[t,p]/4.0/np.pi
      suppression[m2,index,m] += 3 * a2 /mfp_bulk[m]*direction_ave[t,p]*sym*domega[t,p]/4.0/np.pi

   direction = direction_ave.reshape((n_theta * n_phi,3))
   F = np.einsum('m,dj->mdj',mfp_sampled,direction)

   #this is for Fourier----
   angular_average = np.zeros((3,3))
   for t in range(n_theta): 
    for p in range(n_phi):
     angular_average += np.einsum('i,j->ij',direction_ave[t,p],direction_ave[t,p])*domega[t,p]/4.0/np.pi

   rhs_average = np.einsum('m,ij->mij',mfp_sampled*mfp_sampled,angular_average)
   #-----------------------------

   #replicate bulk values---
   kappa = np.zeros((3,3))
   for t in range(n_theta): 
     for p in range(n_phi): 
      for m in range(n_mfp): 
        index = t * n_phi + p 
        tmp = kappa_directional[m,index]
        kappa += np.outer(tmp,direction_ave[t,p])*mfp_sampled[m]

   tc = temp_coeff/np.sum(temp_coeff)


  

   return {'tc':tc,\
           'VMFP':direction,\
           'sigma':kappa_directional,\
           'kappa':kappa,\
           'mfp_average':rhs_average*1e18,\
           'mfp_sampled':mfp_sampled,\
           'suppression':suppression,\
           'kappam':kappa_bulk,\
           'mfp_bulk':mfp_bulk}


