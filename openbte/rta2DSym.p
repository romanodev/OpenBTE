import sys
import numpy as np
import scipy
from .utils import *
import time




def rta_no_interpolation(**argv):


    data = load_data(argv.setdefault('basename','rta'))

    mfp_0 = 1e-10
    mfp_max = 1e-4
    mfp_bulk = np.einsum('ki,k->ki',data['v'],data['tau']) #the minimum MFP is calculated on the real MFP
    I = np.where(np.linalg.norm(mfp_bulk,axis=1) > mfp_0)[0]
    tau = data['tau'][I]
    v = data['v'][I]
    C = data['C'][I]
    f = np.divide(np.ones_like(tau),tau, out=np.zeros_like(tau), where=tau!=0)
    kappam = np.einsum('u,u,u,u->u',tau,C,v[:,0],v[:,0]) 
    Wdiag = C*f
    Jc = np.einsum('k,ki->ki',C,v[:,:2])
    nm = len(Wdiag)

    return  {  'Wdiag':Wdiag,\
               'sigma':Jc,\
               'kappa':data['kappa'],\
               'model':np.array([11])}


def mg2DSym(**argv):

 if not argv.setdefault('interpolation',True):   

     return rta_no_interpolation(**argv)

 #Import data----
 #Get options----
 n_phi = int(argv.setdefault('n_phi',48))
 n_mfp = int(argv.setdefault('n_mfp',50))
 n_theta = 48

 nm = n_phi * n_mfp

 #Create sampled MFPs
 #Polar Angle---------
 Dphi = 2*np.pi/n_phi
 phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
 #phi = np.linspace(0,2.0*np.pi,n_phi,endpoint=False)
 polar_ave = np.array([np.sin(phi),np.cos(phi),np.zeros(n_phi)]).T

 
 #Import data-----------
 data = load_data(argv.setdefault('filename','rta'))

 #small cut on MFPs
 mfp_0 = 1e-10
 mfp_bulk = np.einsum('ki,k->ki',data['v'],data['tau']) #the minimum MFP is calculated on the real MFP
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
 mfp_bulk = np.einsum('ki,k->ki',v[:,:2],tau) #here we have the 2D one
 r = np.linalg.norm(mfp_bulk,axis=1)
 phi_bulk = np.array([np.arctan2(m[0],m[1]) for m in mfp_bulk])
 phi_bulk[np.where(phi_bulk < 0) ] = 2*np.pi + phi_bulk[np.where(phi_bulk <0)]
 kappa = data['kappa']

 mfp_max = argv.setdefault('mfp_max',np.max(mfp_bulk)*10)
 mfp_min = argv.setdefault('mfp_min',mfp_0)
 mfp_sampled = np.logspace(np.log10(mfp_min)*1.01,np.log10(mfp_max),n_mfp,endpoint=True)#min MFP = 1e-1 nm

 
 #-----------------------
 n_mfp_bulk = len(mfp_bulk) 

 temp_coeff = np.zeros((n_mfp,n_phi))
 kappa_directional = np.zeros((n_mfp,n_phi,2)) 

 lo = 0

 #Interpolation in the MFPs
 r_cut = r[r>mfp_0]
 tc_cut = tc[r>mfp_0]
 Jc_cut = Jc[r>mfp_0]
 phi_cut = phi_bulk[r>mfp_0]

 #Interpolation in the MFPs
 a1,a2,m1,m2 = fast_interpolation(r_cut,mfp_sampled)

 #Interpolation in phi---
 b1,b2,p1,p2 = fast_interpolation(phi_cut,phi,bound='periodic')

 #CHECKED
 np.add.at(temp_coeff,(m1, p1),a1*b1*tc_cut)
 np.add.at(temp_coeff,(m1, p2),a1*b2*tc_cut)
 np.add.at(temp_coeff,(m2, p1),a2*b1*tc_cut)
 np.add.at(temp_coeff,(m2, p2),a2*b2*tc_cut)

 #CHECKED 
 np.add.at(kappa_directional,(m1, p1),Jc_cut*(a1*b1)[:,np.newaxis])
 np.add.at(kappa_directional,(m1, p2),Jc_cut*(a1*b2)[:,np.newaxis])
 np.add.at(kappa_directional,(m2, p1),Jc_cut*(a2*b1)[:,np.newaxis])
 np.add.at(kappa_directional,(m2, p2),Jc_cut*(a2*b2)[:,np.newaxis])

 lo_cut = np.sum(tc[r<mfp_0])
 
 temp_coeff *=(1+lo_cut)

 assert(abs(np.sum(temp_coeff)) -1 < 1e-5)
 #Adjusment for numerical stability temp_coeff /= np.sum(temp_coeff)

 #MG---

 print(np.shape(temp_coeff))

 
 

 quit()



 rhs_average = mfp_sampled*mfp_sampled/2

 output =  {'tc':temp_coeff,\
         'sigma':kappa_directional,\
         'kappa':kappa,\
         'mfp_average':rhs_average*1e18,\
         'VMFP':polar_ave[:,:2],\
         'mfp_sampled':mfp_sampled,\
         'sampling': np.array([n_phi,n_theta,n_mfp]),\
         'model':np.array([8])}



 return output
