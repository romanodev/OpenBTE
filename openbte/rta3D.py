import sys
import numpy as np
import scipy
from .utils import *


def generate_rta3D(**argv):

 #Import data----
 #Get options----

 #Compute directions---------------
 n_phi = int(argv.setdefault('n_phi',48))
 n_theta = int(argv.setdefault('n_phi',24))
 n_mfp = argv.setdefault('n_mfp',50)
 Dphi = 2.0*np.pi/n_phi
 phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
 Dtheta = np.pi/n_theta
 theta = np.linspace(0,np.pi,n_theta)
 polar = np.array([np.sin(phi),np.cos(phi),np.ones(n_phi)]).T
 azimuthal = np.array([np.sin(theta),np.sin(theta),np.cos(theta)]).T
 direction = np.einsum('lj,kj->lkj',azimuthal,polar)
 direction = direction.reshape((n_theta * n_phi,3))
 #direction = np.concatenate((direction,np.array([[0,0,1]])),axis=0) #nort pole
 #direction = np.concatenate((direction,np.array([[0,0,-1]])),axis=0) #south pole
 #-------------------

 #Import data-----------
 #a = np.load('rta.npz',allow_pickle=True)
 #data = {key:a[key].item() for key in a}['arr_0']

 data = load_data('rta')

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

 #We perform three separate linear interpolations
 for m in range(n_mfp_bulk):

     (m1,a1,m2,a2) = interpolate(mfp_sampled,r[m],bounds='extent')
     (p1,b1,p2,b2) = interpolate(phi,phi_bulk[m],bounds='periodic',period=2*np.pi)
     (t1,c1,t2,c2) = interpolate(theta,theta_bulk[m],bounds='periodic')

     index_1 =  t1 * n_phi + p1; u1 = b1*c1
     index_2 =  t1 * n_phi + p2; u2 = b2*c1
     temp_coeff[m1,index_1] += tc[m]*a1*u1
     temp_coeff[m1,index_2] += tc[m]*a1*u2
     temp_coeff[m2,index_1] += tc[m]*a2*u1
     temp_coeff[m2,index_2] += tc[m]*a2*u2
     kappa_directional[m1,index_1] += Jc[m]*a1*u1
     kappa_directional[m1,index_2] += Jc[m]*a1*u2
     kappa_directional[m2,index_1] += Jc[m]*a2*u1
     kappa_directional[m2,index_2] += Jc[m]*a2*u2

     index_3 =  t2 * n_phi + p1; u3 = b1*c2
     index_4 =  t2 * n_phi + p2; u4 = b2*c2
     temp_coeff[m1,index_3] += tc[m]*a1*u3
     temp_coeff[m1,index_4] += tc[m]*a1*u4
     temp_coeff[m2,index_3] += tc[m]*a2*u3
     temp_coeff[m2,index_4] += tc[m]*a2*u4
     kappa_directional[m1,index_3] += Jc[m]*a1*u3
     kappa_directional[m1,index_4] += Jc[m]*a1*u4
     kappa_directional[m2,index_3] += Jc[m]*a2*u3
     kappa_directional[m2,index_4] += Jc[m]*a2*u4

 rhs_average = mfp_sampled*mfp_sampled/3

 return {'tc':temp_coeff,\
         'sigma':kappa_directional,\
         'kappa':kappa,\
         'mfp_average':rhs_average*1e18,\
         'VMFP':direction,\
         'mfp_sampled':mfp_sampled,\
         'model':np.array([9])}


