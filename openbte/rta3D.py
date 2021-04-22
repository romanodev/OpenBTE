import sys
import numpy as np
import scipy
from .utils import *
import time

def rta3D(**argv):


 #Compute directions---------------
 n_phi = int(argv.setdefault('n_phi',48))
 n_theta = int(argv.setdefault('n_theta',24))
 n_mfp = argv.setdefault('n_mfp',30)
 Dphi = 2.0*np.pi/n_phi
 phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
 Dtheta = np.pi/n_theta
 theta = np.linspace(Dtheta/2,np.pi-Dtheta/2,n_theta,endpoint=True)

 polar = np.array([np.sin(phi),np.cos(phi),np.ones(n_phi)]).T
 azimuthal = np.array([np.sin(theta),np.sin(theta),np.cos(theta)]).T
 direction = np.einsum('lj,kj->lkj',azimuthal,polar)
 direction = direction.reshape((n_theta * n_phi,3))
 #-------------------

 #Import data-----------

 data = load_data(argv.setdefault('filename','rta'))


 f = np.divide(np.ones_like(data['tau']), data['tau'], out=np.zeros_like(data['tau']), where=data['tau']!=0)
 kappa = data['kappa']
 #--------------------------

 mfp_bulk = np.einsum('ki,k->ki',data['v'],data['tau'])
 r = np.array([np.linalg.norm(m) for m in mfp_bulk])
 #Eliminate zero MFP modes

 mfp_bulk = mfp_bulk[r > 0]
 C = data['C'][r>0] 
 f = f[r>0] 
 v = data['v'][r>0] 
 tc = C*f
 tc /= tc.sum()
 Jc = np.einsum('k,ki->ki',C,v)

 r = np.array([np.linalg.norm(m) for m in mfp_bulk])

 phi_bulk = np.array([np.arctan2(m[0],m[1]) for m in mfp_bulk])
 phi_bulk[np.where(phi_bulk < 0) ] = 2*np.pi + phi_bulk[np.where(phi_bulk <0)]
 theta_bulk = np.array([np.arccos((m/r[k])[2]) for k,m in enumerate(mfp_bulk)])
 mfp_sampled = np.logspace(-10,np.log10(max(r)*1.01),n_mfp)#min MFP = 1e-1 nm

 #-----------------------
 n_mfp_bulk = len(mfp_bulk) 
 mfp = np.logspace(-1,np.log10(max(r)*1.01),n_mfp)#min MFP = 1e-2 
 n_mfp = len(mfp)
 temp_coeff = np.zeros((n_mfp,n_phi*n_theta))
 kappa_directional = np.zeros((n_mfp,n_phi*n_theta,3)) 

 a = time.time()
 #NEW---
 a1,a2,m1,m2 = fast_interpolation(r,mfp_sampled,bound='extent')
 b1,b2,p1,p2 = fast_interpolation(phi_bulk,phi,bound='periodic')
 c1,c2,t1,t2 = fast_interpolation(theta_bulk,theta,bound='extent')

 index_1 =  t1 * n_phi + p1; u1 = b1*c1
 index_2 =  t1 * n_phi + p2; u2 = b2*c1
 index_3 =  t2 * n_phi + p1; u3 = b1*c2
 index_4 =  t2 * n_phi + p2; u4 = b2*c2

 #Interpolate temperature----
 temp_coeff2 = np.zeros((n_mfp,n_phi*n_theta))
 np.add.at(temp_coeff,(m1,index_1),a1*u1*tc)
 np.add.at(temp_coeff,(m1,index_2),a1*u2*tc)
 np.add.at(temp_coeff,(m2,index_1),a2*u1*tc)
 np.add.at(temp_coeff,(m2,index_2),a2*u2*tc)
 np.add.at(temp_coeff,(m1,index_3),a1*u3*tc)
 np.add.at(temp_coeff,(m1,index_4),a1*u4*tc)
 np.add.at(temp_coeff,(m2,index_3),a2*u3*tc)
 np.add.at(temp_coeff,(m2,index_4),a2*u4*tc)

 #Interpolate flux----
 np.add.at(kappa_directional,(m1, index_1),Jc*(a1*u1)[:,np.newaxis])
 np.add.at(kappa_directional,(m1, index_2),Jc*(a1*u2)[:,np.newaxis])
 np.add.at(kappa_directional,(m2, index_1),Jc*(a2*u1)[:,np.newaxis])
 np.add.at(kappa_directional,(m2, index_2),Jc*(a2*u2)[:,np.newaxis])
 np.add.at(kappa_directional,(m1, index_3),Jc*(a1*u3)[:,np.newaxis])
 np.add.at(kappa_directional,(m1, index_4),Jc*(a1*u4)[:,np.newaxis])
 np.add.at(kappa_directional,(m2, index_3),Jc*(a2*u3)[:,np.newaxis])
 np.add.at(kappa_directional,(m2, index_4),Jc*(a2*u4)[:,np.newaxis])

 
 rhs_average = mfp_sampled*mfp_sampled/3

 return {'tc':temp_coeff,\
         'sigma':kappa_directional,\
         'kappa':kappa,\
         'mfp_average':rhs_average*1e18,\
         'sampling': np.array([n_phi,n_theta,n_mfp]),\
         'VMFP':direction,\
         'mfp_sampled':mfp_sampled,\
         'model':np.array([9])}

   
