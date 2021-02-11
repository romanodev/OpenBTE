import sys
import numpy as np
import scipy
from .utils import *
from mpi4py import MPI

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




def mfp3D(**argv): 

   #load data
   if argv.setdefault('read_from_file',True):
     data = load_data('mfp')
     Kacc = data['Kacc']
     mfp_bulk  = data['mfp']
   else:  
     mfp_bulk = argv['mfp']
     Kacc    = argv['Kacc']

   #Get the distribution
   kappa_bulk = Kacc[-1]
   kappa_bulk = np.zeros_like(Kacc)
   kappa_bulk[0] = Kacc[0]



   #Polar Angle-----
   n_phi = int(argv.setdefault('n_phi',48)); Dphi = 2.0*np.pi/n_phi
   #phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
   phi = np.linspace(0,2.0*np.pi,n_phi,endpoint=False)
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


   n_mfp_bulk = len(mfp_bulk)

   if len(mfp_bulk) == n_mfp:
     mfp_sampled = mfp_bulk
   else:  
     mfp_sampled = np.logspace(min([-2,np.log10(min(mfp_bulk)*0.99)]),np.log10(max(mfp_bulk)*1.01),n_mfp)#min MFP = 1e-2 nm 


   n_mfp = len(mfp_sampled)
   temp_coeff = np.zeros(nm) 
   kappa_directional = np.zeros((n_mfp,n_theta*n_phi,3)) 
   kdp = np.zeros((n_mfp,n_theta*n_phi,3)) 
   kd = np.zeros((n_mfp,n_theta*n_phi,3)) 

   #suppression = np.zeros((n_mfp,n_phi*n_theta,n_mfp_bulk)) 
   temp_coeff = np.zeros((n_mfp,n_theta*n_phi))
   tcp = np.zeros((n_mfp,n_theta*n_phi))
   tc = np.zeros((n_mfp,n_theta*n_phi))

   g1 = kappa_bulk/mfp_bulk/mfp_bulk*sym
   g2 = kappa_bulk/mfp_bulk*sym

   n_angles = n_theta*n_phi
   dirr = sym*np.einsum('tpi,tp->tpi',direction_ave,domega).reshape((n_angles,3))
   domega = domega.reshape((n_angles))

   block =  n_mfp_bulk//comm.size
   rr = range(block*comm.rank,n_mfp_bulk) if comm.rank == comm.size-1 else range(block*comm.rank,block*(comm.rank+1))

   mfp_sampled_log = np.log10(mfp_sampled)
   for m in rr:  

    if argv.setdefault('log_interpolation',False):     
     (m1,a1,m2,a2) = get_linear_indexes(mfp_sampled_log,np.log10(mfp_bulk[m]))
    else: 
     (m1,a1,m2,a2) = get_linear_indexes(mfp_sampled,mfp_bulk[m])
   
    kdp[m1] += a1 * g2[m]*dirr
    kdp[m2] += a2 * g2[m]*dirr
    tcp[m1] += a1 * g1[m]*domega
    tcp[m2] += a2 * g1[m]*domega

        
   comm.Allreduce([kdp,MPI.DOUBLE],[kd,MPI.DOUBLE],op=MPI.SUM)
   comm.Allreduce([tcp,MPI.DOUBLE],[tc,MPI.DOUBLE],op=MPI.SUM)

   

   #----
   kd *= 3/(4.0*np.pi)
   tc /= np.sum(tc)
   #----

   direction = direction_ave.reshape((n_theta * n_phi,3))
   F = np.einsum('m,dj->mdj',mfp_sampled,direction)

   rhs_average = mfp_sampled*mfp_sampled/3
   thermal_conductance = 3*mfp_sampled/2
   #-----------------------------

   #replicate bulk values---
   kappa = np.zeros((3,3))
   for t in range(n_theta): 
     for p in range(n_phi): 
      for m in range(n_mfp): 
        index = t * n_phi + p 
        tmp = kd[m,index]
        kappa += np.outer(tmp,direction_ave[t,p])*mfp_sampled[m]



   return {'tc':tc,\
           'VMFP':direction,\
           'sigma':kd,\
           'kappa':kappa,\
           'ang_coeff':domega.reshape(n_phi*n_theta)/4/np.pi,\
           'model':np.array([6]),\
           'mfp_average':rhs_average*1e18,\
           'thermal_conductance':thermal_conductance*1e9,\
           'mfp_sampled':mfp_sampled,\
           'suppression':np.zeros(1),\
           'kappam':np.zeros(1),\
           'mfp_bulk':mfp_bulk}


