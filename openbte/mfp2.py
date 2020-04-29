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
   n_theta = int( argv.setdefault('n_theta',24)  /2)
   #n_theta = int( argv.setdefault('n_theta',24))
   sym = 2

   Dtheta = np.pi/n_theta/2.0
   theta = np.linspace(Dtheta/2.0,np.pi/2.0 - Dtheta/2.0,n_theta)
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
   temp_coeff = np.zeros((n_mfp,n_theta*n_phi))
   kappa_m = np.zeros(n_mfp)
   for m in range(n_mfp_bulk):
    (m1,a1,m2,a2) = get_linear_indexes(mfp_sampled,mfp_bulk[m],scale='linear',extent=True)
    kappa_m[m1] += a1*kappa_bulk[m] 
    kappa_m[m2] += a2*kappa_bulk[m] 
    for t in range(n_theta): 
     for p in range(n_phi): 
      index_1 = t * n_phi + p
      index_2 = t * n_phi + p 
      factor = kappa_bulk[m]/4.0/np.pi*domega[t,p]
      kappa_directional[m1,index_1] += 3 * a1 * factor/mfp_bulk[m]*direction_ave[t,p]*sym
      kappa_directional[m2,index_2] += 3 * a2 * factor/mfp_bulk[m]*direction_ave[t,p]*sym
      temp_coeff[m1,index_1] += a1 * kappa_bulk[m]/mfp_bulk[m]/mfp_bulk[m]*domega[t,p]*sym
      temp_coeff[m2,index_2] += a2 * kappa_bulk[m]/mfp_bulk[m]/mfp_bulk[m]*domega[t,p]*sym

   #kappa = np.zeros((3,3))
   direction = direction_ave.reshape((n_theta * n_phi,3))
   #direction = np.array([np.repeat(direction[:,i],n_mfp) for i in range(3)]).T
   #mfp = np.tile(mfp_sampled,n_phi*n_theta)  
   #F = np.einsum('i,ij->ij',mfp,direction)

   #Build multiscale directions---
   #F = np.einsum('m,dj->mdj',np.flip(mfp_sampled),direction)
   F = np.einsum('m,dj->mdj',mfp_sampled,direction)
   #temp_coeff = np.flip(temp_coeff,axis=0)
   #kappa_directional = np.flip(kappa_directional,axis=0)

   #this is for Fourier----
   angular_average = np.zeros((3,3))
   for t in range(n_theta): 
    for p in range(n_phi):
     angular_average += np.einsum('i,j->ij',direction_ave[t,p],direction_ave[t,p])*domega[t,p]/4.0/np.pi

   #kappa_average = np.einsum('l,ij->lij',3*np.flip(kappa_m),angular_average)
   kappa_average = np.einsum('l,ij->lij',3*kappa_m,angular_average)
   #rhs_average = 3/np.flip(mfp_sampled)/np.flip(mfp_sampled)
   rhs_average = mfp_sampled*mfp_sampled/3
   #-----------------------------

   #replicate bulk values---
   kappa = np.zeros((3,3))
   for t in range(n_theta): 
     for p in range(n_phi): 
      for m in range(n_mfp): 
        index = t * n_phi + p 
        tmp = kappa_directional[m,index]
        kappa += np.outer(tmp,direction_ave[t,p])*mfp_sampled[m]

   #------------------
   #-----------------------------
   tc = temp_coeff/np.sum(temp_coeff)


   return {'temp':tc,'B':[],'F':F,'G':kappa_directional,'kappa':kappa,'scale':np.ones((n_mfp,n_theta*n_phi)),'ac':tc,'mfp_average':rhs_average,'kappa_average':kappa_average}


