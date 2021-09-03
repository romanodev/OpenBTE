import numpy as np 
import openbte.utils as utils 
from mpi4py import MPI

comm = MPI.COMM_WORLD


def rta3D(rta,options_material)->'material':

 #Parse options
 data = None 
 if comm.rank == 0:

  #Parse options
  n_phi = options_material.setdefault('n_phi',48)
  n_mfp = options_material.setdefault('n_mfp',50)
  n_theta = options_material.setdefault('n_theta',24)
  #----------------

  n_angles = n_phi * n_theta
  #Angular discretization
  Dphi = 2.0*np.pi/n_phi
  phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
  Dtheta = np.pi/n_theta
  theta = np.linspace(Dtheta/2,np.pi-Dtheta/2,n_theta,endpoint=True)
  #theta = np.linspace(0,np.pi,n_theta,endpoint=True)
  polar = np.array([np.sin(phi),np.cos(phi),np.ones(n_phi)]).T
  azimuthal = np.array([np.sin(theta),np.sin(theta),np.cos(theta)]).T
  direction_original = np.einsum('lj,kj->lkj',azimuthal,polar)
  direction = direction_original.reshape((n_theta * n_phi,3))
  #-------------------

  #Compute mfp_bulk--
  tau   = rta['tau']
  v     = rta['v']
  C     = rta['C']
  sigma = np.einsum('k,ki->ki',C,v)
  mfp_bulk = np.einsum('ki,k->ki',v,tau) 
  f     = np.divide(np.ones_like(tau),tau, out=np.zeros_like(tau), where=tau!=0)
  Wdiag    = C*f
  r_bulk,phi_bulk,theta_bulk = utils.compute_spherical(mfp_bulk)

  #Sampling
  mfp_max = np.max(r_bulk)*1.1
  mfp_min = np.min(r_bulk)*0.9
  mfp_sampled = np.logspace(np.log10(mfp_min)*1.01,np.log10(mfp_max),n_mfp,endpoint=True)

  Wdiag_sampled = np.zeros((n_mfp,n_angles))
  sigma_sampled = np.zeros((n_mfp,n_angles,3)) 

  a1,a2,m1,m2 = utils.fast_interpolation(r_bulk,mfp_sampled,bound='extent')
  b1,b2,p1,p2 = utils.fast_interpolation(phi_bulk,phi,bound='periodic')
  c1,c2,t1,t2 = utils.fast_interpolation(theta_bulk,theta,bound='extent')

  index_1 =  t1 * n_phi + p1; u1 = b1*c1
  index_2 =  t1 * n_phi + p2; u2 = b2*c1
  index_3 =  t2 * n_phi + p1; u3 = b1*c2
  index_4 =  t2 * n_phi + p2; u4 = b2*c2

  #Temperature coefficients
  np.add.at(Wdiag_sampled,(m1,index_1),a1*u1*Wdiag)
  np.add.at(Wdiag_sampled,(m1,index_2),a1*u2*Wdiag)
  np.add.at(Wdiag_sampled,(m2,index_1),a2*u1*Wdiag)
  np.add.at(Wdiag_sampled,(m2,index_2),a2*u2*Wdiag)
  np.add.at(Wdiag_sampled,(m1,index_3),a1*u3*Wdiag)
  np.add.at(Wdiag_sampled,(m1,index_4),a1*u4*Wdiag)
  np.add.at(Wdiag_sampled,(m2,index_3),a2*u3*Wdiag)
  np.add.at(Wdiag_sampled,(m2,index_4),a2*u4*Wdiag)

  #Interpolate flux----
  np.add.at(sigma_sampled,(m1, index_1),sigma*(a1*u1)[:,np.newaxis])
  np.add.at(sigma_sampled,(m1, index_2),sigma*(a1*u2)[:,np.newaxis])
  np.add.at(sigma_sampled,(m2, index_1),sigma*(a2*u1)[:,np.newaxis])
  np.add.at(sigma_sampled,(m2, index_2),sigma*(a2*u2)[:,np.newaxis])
  np.add.at(sigma_sampled,(m1, index_3),sigma*(a1*u3)[:,np.newaxis])
  np.add.at(sigma_sampled,(m1, index_4),sigma*(a1*u4)[:,np.newaxis])
  np.add.at(sigma_sampled,(m2, index_3),sigma*(a2*u3)[:,np.newaxis])
  np.add.at(sigma_sampled,(m2, index_4),sigma*(a2*u4)[:,np.newaxis])
  #-----------------------

  VMFP = np.einsum('m,ptj->mptj',mfp_sampled,direction_original)
  Wdiag_inv = np.divide(1, Wdiag_sampled, out=np.zeros_like(Wdiag_sampled), where=Wdiag_sampled!=0)
  kappa_sampled = np.einsum('mqi,mq,mqj->ij',sigma_sampled,Wdiag_inv,sigma_sampled)

  #print(kappa_sampled)
  data = {}
  data['sigma'] = sigma_sampled
  data['kappa'] = kappa_sampled
  data['tc']    = Wdiag_sampled/np.sum(Wdiag_sampled)
  data['r_bulk'] = r_bulk
  data['F'] = VMFP
  data['VMFP'] = direction
  data['phi_bulk'] = phi_bulk
  data['phi'] = phi
  data['theta'] = theta
  data['theta_bulk'] = theta_bulk
  data['mfp_bulk'] = mfp_bulk
  data['mfp_sampled'] = mfp_sampled
  data['sigma_bulk'] = sigma
  data['f'] = rta['f']

 return utils.create_shared_memory_dict(data)

