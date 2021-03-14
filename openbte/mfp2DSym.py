import sys
import numpy as np
import scipy
from mpi4py import MPI
import time
import os 
from openbte.utils import *


comm = MPI.COMM_WORLD

def get_linear_indexes(mfp,value):


   n = len(mfp)

   if value < mfp[0]:
      aj = (value-mfp[0])/(mfp[1]-mfp[0]) 
      return 0,1-aj,1,aj  
   elif value > mfp[-1]:
      aj = (value-mfp[n-2])/(mfp[n-1]-mfp[n-2]) 
      return n-2,1-aj,n-1,aj  
   else:
    for m in range(n-1):
      if (value <= mfp[m+1]) and (value >= mfp[m]) :
        aj = (value-mfp[m])/(mfp[m+1]-mfp[m]) 
        return m,1-aj,m+1,aj  
   



def mfp2DSym(**argv):

 #load data
 
 data = load_data('mfp')

 Kacc = data['Kacc']
 mfp_bulk  = data['mfp']

 kappa_bulk = Kacc[-1]
 kappa_bulk = np.zeros_like(Kacc)
 kappa_bulk[0] = Kacc[0]
 for n in range(len(Kacc)-1):
    kappa_bulk[n+1] = Kacc[n+1]-Kacc[n]

 #pruning--
 I = np.where(mfp_bulk > 0)
 kappa_bulk = kappa_bulk[I]
 mfp_bulk = mfp_bulk[I]
 kappa= np.eye(3) * np.sum(kappa_bulk)

 #Get options----
 n_phi = int(argv.setdefault('n_phi',48))
 n_mfp = int(argv.setdefault('n_mfp',50))
 n_theta = int(argv.setdefault('n_theta',24))

 nm = n_phi * n_mfp

 #Create sampled MFPs
 n_mfp_bulk = len(mfp_bulk)
 mfp_sampled = np.logspace(min([-2,np.log10(min(mfp_bulk)*0.99)]),np.log10(max(mfp_bulk)*1.01),n_mfp)#min MFP = 1e-2 nm 

 #Polar Angle---------
 Dphi = 2*np.pi/n_phi
 #phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
 phi = np.linspace(0,2.0*np.pi,n_phi,endpoint=False)
 #--------------------
   
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

 mfp = np.logspace(np.log10(max([min(mfp_bulk),1e-11])),np.log10(max(mfp_bulk)*1.01),n_mfp) 

 n_mfp = len(mfp)

 temp_coeff_p,temp_coeff = np.zeros((2,n_mfp,n_phi))
 kappa_directional_p,kappa_directional = np.zeros((2,n_mfp,n_phi,3)) 
 #suppression_p,suppression = np.zeros((2,n_mfp,n_phi,n_mfp_bulk)) 


 dirr = np.einsum('tpi,tp->tpi',direction_ave[:,:,:2],domega)

  
 kdp = np.zeros((n_mfp,n_phi,2))
 kd = np.zeros((n_mfp,n_phi,2))
 tcp = np.zeros((n_mfp,n_phi))
 tc = np.zeros((n_mfp,n_phi))
 p = np.arange(n_phi)


 g1 = kappa_bulk/mfp_bulk
 g2 = kappa_bulk/mfp_bulk/mfp_bulk
 g3 = ftheta*np.sin(theta)
 n_tot = n_mfp_bulk

 block =  n_tot//comm.size
 rr = range(block*comm.rank,n_tot) if comm.rank == comm.size-1 else range(block*comm.rank,block*(comm.rank+1))

 #change to mfp

 for t in range(n_theta):

   for m in rr:  
       
      (m1,a1,m2,a2) = get_linear_indexes(mfp,mfp_bulk[m]*g3[t])

      tmp = g1[m]*dirr[t]
      kdp[m1] += a1*tmp
      kdp[m2] += a2*tmp

      #Temperature
      tmp = g2[m]*domega[t]
      tcp[m1] += a1*tmp
      tcp[m2] += a2*tmp

      #Suppression
      #tmp  = dirr[t,:,0]/mfp_bulk[m]
      #sp[m1,:,m] +=  a1 * tmp
      #sp[m2,:,m] +=  a2 * tmp

   comm.Allreduce([kdp,MPI.DOUBLE],[kd,MPI.DOUBLE],op=MPI.SUM)
   comm.Allreduce([tcp,MPI.DOUBLE],[tc,MPI.DOUBLE],op=MPI.SUM)
   #comm.Allreduce([sp,MPI.DOUBLE],[s,MPI.DOUBLE],op=MPI.SUM)

 kd *= 2*3/4.0/np.pi
 #s *= 3*2/4.0/np.pi
 tc /= np.sum(tc)


 #Test for bulk---

 if comm.rank == 0:
  kappa_bulk = np.zeros((2,2))
  for m in range(n_mfp):
   for p in range(n_phi):
      kappa_bulk += mfp[m]*np.outer(kd[m,p],polar_ave[p,:2])


 #replicate bulk values---

 #Wod = np.tile(tc,(nm,1))

 #angle_map = np.arange(n_phi)
 #angle_map = np.repeat(angle_map,n_mfp)

 #repeat angle---
 #kappa_directional[:,:,2] = 0 #Enforce zeroflux on z (for visualization purposed)
 #F = np.einsum('m,pi->mpi',mfp,polar_ave)
 rhs_average = mfp_sampled*mfp_sampled/2
 a = np.zeros((3,3)) 
 for i in range(len(polar_ave)):
  a += np.outer(polar_ave[i],polar_ave[i])/2/np.pi*2*np.pi/n_phi
 rhs_average = mfp_sampled*mfp_sampled
 rhs_average *= a[0,0]
 
 #Final----
 return {'tc':tc,\
         'sigma':kd,\
         'kappa':kappa,\
         'mfp_average':rhs_average*1e18,\
         'VMFP':polar_ave[:,:2],\
         'mfp_sampled':mfp,\
         'sampling': np.array([n_phi,n_theta,n_mfp]),\
         'model':np.array([5]),\
         'suppression':np.zeros(1),\
         'kappam':kappa_bulk}


