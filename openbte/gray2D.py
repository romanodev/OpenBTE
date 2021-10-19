import sys
import numpy as np
import openbte.utils as utils
from mpi4py import MPI
comm = MPI.COMM_WORLD
def gray2D(options_material)->'material':

 data = None

 if comm.rank == 0:

    #Parse options
    n_phi = options_material.setdefault('n_phi',48)
    kappa_bulk = options_material['kappa']
    kappa = options_material['kappa']*np.eye(2)
    mfp_bulk = np.array([options_material['mfp']])

    #Import data----
    Dphi = 2*np.pi/n_phi
    phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
    polar = np.array([np.sin(phi),np.cos(phi)]).T
    fphi= np.sinc(Dphi/2.0/np.pi)
    polar_ave = polar*fphi
    #------------------

    sigma_sampled = np.zeros((1,n_phi,2))
    temp_coeff = np.zeros((1,n_phi))
    temp_coeff[0,:] = 1/n_phi
    suppression = np.zeros((1,n_phi,1)) 
    for p in range(n_phi): 
       sigma_sampled[0,p] += kappa_bulk/mfp_bulk * polar_ave[p]/n_phi*2

    VMFP  = np.einsum('m,qi->mqi',mfp_bulk,polar_ave)
    kappa_sampled = np.einsum('mpi,mpj->ij',VMFP,sigma_sampled)

    tc = temp_coeff/np.sum(temp_coeff)

    data = {}
    data['sigma'] = sigma_sampled
    data['kappa'] = kappa_sampled
    data['tc']    = tc
    data['r_bulk'] = mfp_bulk
    data['F'] = VMFP
    data['VMFP'] = polar_ave
    data['phi_bulk'] = phi
    data['phi'] = phi
    data['mfp_bulk'] = mfp_bulk
    data['mfp_sampled'] = mfp_bulk
    data['sigma_bulk'] = sigma_sampled

 return utils.create_shared_memory_dict(data)



