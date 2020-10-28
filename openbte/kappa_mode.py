from openbte import Geometry, Solver, Material, Plot
import numpy as np
from openbte.utils import *



def main():
 #-----------------------
 mat = load_data('material')
 #data = load_data('rta')
 data = mat['origin']
 mfp_0 = 1e-9
 #I = np.where(np.linalg.norm(mfp_bulk,axis=1) > mfp_0)[0]
 I = np.arange(len(data['tau']))
 tau = data['tau'][I]
 v = data['v'][I]
 C = data['C'][I]
 f = data['f'][I]
 #f = np.divide(np.ones_like(tau),tau, out=np.zeros_like(tau), where=tau!=0)
 kappam = np.einsum('u,u,u,u->u',tau,C,v[:,0],v[:,0]) 
 Jc = np.einsum('k,ki->ki',C,v[:,:2])
 nm = len(C)
 mfp_bulk = np.einsum('ki,k->ki',v,tau)
 r = np.linalg.norm(mfp_bulk[:,:2],axis=1)
 phi_bulk = np.array([np.arctan2(m[0],m[1]) for m in mfp_bulk[:,:2]])
 phi_bulk[np.where(phi_bulk < 0) ] = 2*np.pi + phi_bulk[np.where(phi_bulk <0)]
 kappa = data['kappa']
 #---------------------------

 #Compute MFPs
 mat = load_data('material')
 mfp_sampled  = mat['mfp_sampled']*1e9
 [n_phi,n_theta,n_mfp]  = mat['sampling']
 vmfp  = mat['VMFP']
 sigma  = mat['sigma']
 sigma = sigma[:,:,0].T
 Dphi = 2*np.pi/n_phi
 phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
 phi = list(phi)
 mfp_vec = np.outer(vmfp[:,0],mfp_sampled)
 mfp = np.log10(mfp_sampled)
 mfp -= np.min(mfp)
 #--------------------------------


 #Compute kappa
 sol = load_data('solver')
 T_sampled = sol['kappa_mode']

 kappa_fourier = sol['kappa_fourier']

 #Interpolation to get kappa mode
 ratio = kappa_fourier/kappa[0,0]

 T_mode = np.zeros_like(kappam)
 for m in range(nm):

     (m1,a1,m2,a2) = interpolate(mfp_sampled,r[m]*1e9,bounds='extent')
     (p1,b1,p2,b2) = interpolate(phi,phi_bulk[m],bounds='periodic',period=2*np.pi)

     T_mode[m] += T_sampled[m1,p1]*a1*b1
     T_mode[m] += T_sampled[m1,p2]*a1*b2
     T_mode[m] += T_sampled[m2,p1]*a2*b1
     T_mode[m] += T_sampled[m2,p2]*a2*b2

 T_mode *=1e9

 kappa_nano = np.einsum('i,i,i->i',C,v[:,0],T_mode)
 kappa_bulk = np.einsum('i,i,i->i',C,v[:,0],mfp_bulk[:,0])

 kappa_nano_ori = np.einsum('ij,ji',sigma,T_sampled)*1e9

 print('Check:')
 print('kappa (bulk):  ' + str(round(np.sum(kappa_bulk),2)) + ' W/m/K')
 print('kappa (mfp-sampled):  ' + str(round(np.sum(kappa_nano),2)) + ' W/m/K')
 print('kappa (mode-sampled): ' + str(round(kappa_nano_ori,2)) + ' W/m/K')

 mfp_bulk[:,0] = T_mode

 mfp_nano = np.linalg.norm(mfp_bulk,axis=1)

 save_data('suppression',{'kappa_nano':kappa_nano/ratio,'mfp_nano':mfp_nano,'kappa_bulk':kappa_bulk,'mfp_bulk':r,'f':f})


if __name__ == "__main__":

 main()
