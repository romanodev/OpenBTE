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
   #n_theta = int( argv.setdefault('n_theta',24)  /2)
   n_theta = int( argv.setdefault('n_theta',24))
   sym = 1

   Dtheta = np.pi/n_theta
   theta = np.linspace(Dtheta/2.0,np.pi - Dtheta/2.0,n_theta)
   dtheta = 2.0*np.sin(Dtheta/2.0)*np.sin(theta)


   #Compute directions---
   polar = np.array([np.sin(phi),np.cos(phi),np.ones(n_phi)]).T
   azimuthal = np.array([np.sin(theta),np.sin(theta),np.cos(theta)]).T
   direction = np.einsum('ij,kj->ikj',azimuthal,polar)

   #Compute average---
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

   #direction_int = []
   control_angle = []
   for t in range(n_theta): 
    for p in range(n_phi): 
     for m in range(n_mfp):
      control_angle.append(direction_ave[t,p,:])

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
   kappa_directional = np.zeros((nm,3)) 

   for m in range(n_mfp_bulk):
    (m1,a1,m2,a2) = get_linear_indexes(mfp_sampled,mfp_bulk[m],scale='linear',extent=True)
    for t in range(n_theta): 
     for p in range(n_phi): 
      index_1 = t * n_mfp * n_phi + p * n_mfp + m1
      index_2 = t * n_mfp * n_phi + p * n_mfp + m2
      factor = kappa_bulk[m]/4.0/np.pi*domega[t,p]
          
      kappa_directional[index_1] += 3 * a1 * factor/mfp_bulk[m]*direction_ave[t,p]*sym
      kappa_directional[index_2] += 3 * a2 * factor/mfp_bulk[m]*direction_ave[t,p]*sym
      temp_coeff[index_1] += a1 * kappa_bulk[m]/mfp_bulk[m]/mfp_bulk[m]*domega[t,p]*sym
      temp_coeff[index_2] += a2 * kappa_bulk[m]/mfp_bulk[m]/mfp_bulk[m]*domega[t,p]*sym


   #replicate bulk values---
   kappa = np.zeros((3,3))
   for t in range(n_theta): 
     for p in range(n_phi): 
      for m in range(n_mfp): 
        index = t * n_mfp * n_phi + p * n_mfp + m
        tmp = kappa_directional[index]
        kappa += np.outer(tmp,direction_ave[t,p])*mfp_sampled[m]
   #------------------


   #kappa = np.zeros((3,3))
   direction = direction_ave.reshape((n_theta * n_phi,3))
   direction = np.array([np.repeat(direction[:,i],n_mfp) for i in range(3)]).T
   mfp = np.tile(mfp_sampled,n_phi*n_theta)  
   F = np.einsum('i,ij->ij',mfp,direction)

   #Build multiscale directions---

   

   #-----------------------------



   #for g,f in zip(kappa_directional,F):
   #   kappa += np.outer(g,f) 
  
   tc = temp_coeff/np.sum(temp_coeff)


   return {'temp':tc,'B':[],'F':F,'G':kappa_directional,'kappa':kappa,'scale':np.ones(nm),'ac':tc}


