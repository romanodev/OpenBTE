import sys
import numpy as np
import scipy
from .utils import *
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD


def generate_rta2DSym(**argv):

 #Import data----
 #Get options----
 n_phi = int(argv.setdefault('n_phi',48))
 n_mfp = int(argv.setdefault('n_mfp',50))
 n_theta = 24

 nm = n_phi * n_mfp

 #Create sampled MFPs
 #Polar Angle---------
 Dphi = 2*np.pi/n_phi
 #phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
 phi = np.linspace(0,2.0*np.pi-Dphi,n_phi,endpoint=True)

 polar_ave = np.array([np.sin(phi),np.cos(phi),np.zeros(n_phi)]).T

 #Import data-----------
 
 #data = dd.io.load('rta.h5')

 data = load_data('rta')

 #kappa = np.einsum('k,ki,kj,k->ij',data['C'],data['v'],data['v'],data['tau'])
 #small cut on MFPs
 mfp_0 = 1e-9
 mfp_bulk = np.einsum('ki,k->ki',data['v'][:,:2],data['tau'])
 I = np.where(np.linalg.norm(mfp_bulk,axis=1) > mfp_0)[0]

 tau = data['tau'][I]
 v = data['v'][I]
 C = data['C'][I]

 f = np.divide(np.ones_like(tau),tau, out=np.zeros_like(tau), where=tau!=0)
 kappam = np.einsum('u,u,u,u->u',tau,C,v[:,0],v[:,0]) 
 tc = C*f
 tc /= tc.sum()
 Jc = np.einsum('k,ki->ki',C,v[:,:2])
 nm = len(tc)
 mfp_bulk = np.einsum('ki,k->ki',v[:,:2],tau)
 r = np.linalg.norm(mfp_bulk,axis=1)
 phi_bulk = np.array([np.arctan2(m[0],m[1]) for m in mfp_bulk])
 phi_bulk[np.where(phi_bulk < 0) ] = 2*np.pi + phi_bulk[np.where(phi_bulk <0)]
 kappa = data['kappa']

 mfp_sampled = np.logspace(np.log10(mfp_0)*1.01,np.log10(max(r)*1.01),n_mfp,endpoint=True)#min MFP = 1e-1 nm

 #mfp_sampled = np.logspace(-10,-1,n_mfp)#min MFP = 1e-1 nm
 
 #-----------------------
 n_mfp_bulk = len(mfp_bulk) 

 temp_coeff = np.zeros((n_mfp,n_phi))
 kappa_directional = np.zeros((n_mfp,n_phi,2)) 
 #suppression = np.zeros((n_mfp,n_phi,n_mfp_bulk)) 

 interp = np.zeros((n_mfp,n_phi))
 lo = 0
 a = time.time()



 #block =  nm//comm.size
 #rr = range(block*comm.rank,nm) if comm.rank == comm.size-1 else range(block*comm.rank,block*(comm.rank+1))
 

 #tc,tcp = np.zeros((2,n_mfp,n_phi))
 #kc,kcp = np.zeros((2,n_mfp,n_phi,2))

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
 #comm.Allreduce([kdp,MPI.DOUBLE],[kd,MPI.DOUBLE],op=MPI.SUM)
 #comm.Allreduce([tcp,MPI.DOUBLE],[tc,MPI.DOUBLE],op=MPI.SUM)

 
 temp_coeff *=(1+lo)

 rhs_average = mfp_sampled*mfp_sampled/2

 return {'tc':temp_coeff,\
         'sigma':kappa_directional,\
         'kappa':kappa,\
         'mfp_average':rhs_average*1e18,\
         'VMFP':polar_ave[:,:2],\
         'mfp_sampled':mfp_sampled,\
         'sampling': np.array([n_phi,n_theta,n_mfp]),\
         'model':np.array([8])}


