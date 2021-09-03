import numpy as np 
import openbte.utils as utils 
from mpi4py import MPI

comm = MPI.COMM_WORLD

def rta2DSym(rta,options_material)->'material':

   #Parse options
   data = None 
   if comm.rank == 0:
    n_phi = options_material.setdefault('n_phi',48)
    n_mfp = options_material.setdefault('n_mfp',50)
    nm = n_phi * n_mfp
    Dphi = 2*np.pi/n_phi
    phi = np.linspace(Dphi/2,2.0*np.pi-Dphi/2,n_phi,endpoint=True)
    polar_ave = np.array([np.sin(phi),np.cos(phi)]).T

    #Compute mfp_bulk--
    tau   = rta['tau']
    v     = rta['v']
    C     = rta['C']
    sigma = np.einsum('k,ki->ki',C,v[:,:2])
    mfp_bulk = np.einsum('ki,k->ki',v[:,:2],tau) #here we have the 2D one
    f     = np.divide(np.ones_like(tau),tau, out=np.zeros_like(tau), where=tau!=0)
    Wdiag    = C*f
    r_bulk,phi_bulk = utils.compute_polar(mfp_bulk)

    #Sampling
    mfp_max = np.max(r_bulk)*1.1
    mfp_min = np.min(r_bulk)*0.9
    mfp_sampled = np.logspace(np.log10(mfp_min)*1.01,np.log10(mfp_max),n_mfp,endpoint=True)
    #-----------------

    Wdiag_sampled = np.zeros((n_mfp,n_phi))
    sigma_sampled = np.zeros((n_mfp,n_phi,2)) 

    #Interpolation in the MFPs
    a1,a2,m1,m2 = utils.fast_interpolation(r_bulk,mfp_sampled,bound='extent')
    #Interpolation in phi---
    b1,b2,p1,p2 = utils.fast_interpolation(phi_bulk,phi,bound='periodic')
    
    np.add.at(Wdiag_sampled,(m1, p1),a1*b1*Wdiag)
    np.add.at(Wdiag_sampled,(m1, p2),a1*b2*Wdiag)
    np.add.at(Wdiag_sampled,(m2, p1),a2*b1*Wdiag)
    np.add.at(Wdiag_sampled,(m2, p2),a2*b2*Wdiag)

    #lo_cut = np.sum(tc[r<mfp_0])
    #Wdiag_sampled *=(1+lo_cut)

    np.add.at(sigma_sampled,(m1, p1),sigma*(a1*b1)[:,np.newaxis])
    np.add.at(sigma_sampled,(m1, p2),sigma*(a1*b2)[:,np.newaxis])
    np.add.at(sigma_sampled,(m2, p1),sigma*(a2*b1)[:,np.newaxis])
    np.add.at(sigma_sampled,(m2, p2),sigma*(a2*b2)[:,np.newaxis])

    VMFP = np.einsum('m,qj->mqj',mfp_sampled,polar_ave[:,:2])

    Wdiag_inv = np.divide(1, Wdiag_sampled, out=np.zeros_like(Wdiag_sampled), where=Wdiag_sampled!=0)
    kappa_sampled = np.einsum('mqi,mq,mqj->ij',sigma_sampled,Wdiag_inv,sigma_sampled)
    #print(kappa_sampled)
    #kappa_sampled = np.einsum('mqi,mqj->ij',VMFP,sigma_sampled)/rta['alpha'][0][0] #Second method

    #VMFP = np.einsum('m,qj->mqj',mfp_sampled,polar_ave[:,:2]).reshape(nm,2)
    data = {}
    data['sigma'] = sigma_sampled
    data['kappa'] = kappa_sampled
    data['tc']    = Wdiag_sampled/np.sum(Wdiag_sampled)
    data['r_bulk'] = r_bulk
    data['F'] = VMFP
    data['VMFP'] = polar_ave
    data['phi_bulk'] = phi_bulk
    data['phi'] = phi
    data['mfp_bulk'] = mfp_bulk
    data['mfp_sampled'] = mfp_sampled
    data['sigma_bulk'] = sigma
    data['f'] = rta['f']

   return utils.create_shared_memory_dict(data)


