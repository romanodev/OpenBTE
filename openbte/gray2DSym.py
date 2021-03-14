import sys
import numpy as np
from .utils import *


def gray2DSym(**argv):

 #Import data----
 #kappa_bulk = argv['kappa']
 #kappa = argv['kappa']*np.eye(2)
 #mfp_bulk = argv['mfp']
 #n_phi = int(argv.setdefault('n_phi',48)) 
 #n_mfp = int(argv.setdefault('n_mfp',50)) 
 #n_theta = int(argv.setdefault('n_theta',48))
 #Dphi = 2*np.pi/n_phi
 #phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
 #polar = np.array([np.sin(phi),np.cos(phi)]).T
 #Dtheta = np.pi/n_theta
 #theta = np.linspace(Dtheta/2.0,np.pi - Dtheta/2.0,n_theta)
 #dtheta = 2.0*np.sin(Dtheta/2.0)*np.sin(theta)   
 #domega = np.outer(dtheta,Dphi * np.ones(n_phi))
 #ftheta        = (1-np.cos(2*theta)*np.sinc(Dtheta/np.pi))/(np.sinc(Dtheta/2/np.pi)*(1-np.cos(2*theta)))
 #fphi          = np.sinc(Dphi/2.0/np.pi)
 #polar_ave     = polar*fphi
 #azimuthal_ave = np.array([ftheta*np.sin(theta),ftheta*np.sin(theta),np.cos(theta)]).T  
 #direction_ave = np.einsum('tj,pj->tpj',azimuthal_ave,polar_ave)
 #direction_int = np.einsum('ijl,ij->ijl',direction_ave,domega)
 #------------------------------------------------------

 kappa_bulk = argv['kappa']
 kappa = argv['kappa']*np.eye(3)
 mfp_bulk = argv['mfp']
 #kappa_bulk = np.eye(3)
 n_phi = int(argv.setdefault('n_phi',48)) 
 n_mfp = int(argv.setdefault('n_mfp',50)) #this is K
 n_theta = int(argv.setdefault('n_theta',48))
 nm = n_phi * n_mfp
 #Create sampled MFPs
 min_mfp = 1e-10 #m
 max_mfp = 1e-3  #m
 mfp = np.logspace(np.log10(min_mfp),np.log10(max_mfp),n_mfp)
 n_mfp = len(mfp)
 #Polar Angle---------
 Dphi = 2*np.pi/n_phi
 phi = np.linspace(0,2.0*np.pi,n_phi,endpoint=False)
 #Azimuthal Angle------------------------------
 Dtheta = np.pi/n_theta
 theta = np.linspace(Dtheta/2.0,np.pi - Dtheta/2.0,n_theta)
 dtheta = 2.0*np.sin(Dtheta/2.0)*np.sin(theta)   
 domega = np.outer(dtheta,Dphi * np.ones(n_phi))

 #Compute directions---
 polar = np.array([np.sin(phi),np.cos(phi),np.ones(n_phi)]).T
 azimuthal = np.array([np.sin(theta),np.sin(theta),np.cos(theta)]).T
 direction = np.einsum('tj,pj->tpj',azimuthal,polar)
 
 #Compute average---
 ftheta        = (1-np.cos(2*theta)*np.sinc(Dtheta/np.pi))/(np.sinc(Dtheta/2/np.pi)*(1-np.cos(2*theta)))
 fphi          = np.sinc(Dphi/2.0/np.pi)
 polar_ave     = polar*fphi
 azimuthal_ave = np.array([ftheta*np.sin(theta),ftheta*np.sin(theta),np.cos(theta)]).T  
 direction_ave = np.einsum('tj,pj->tpj',azimuthal_ave,polar_ave)
 direction_int = np.einsum('ijl,ij->ijl',direction_ave,domega)

 #-----------------

 #Gray 2D Sym--------
 n_tot = n_phi*n_theta
 kappa_directional = np.zeros((1,n_tot,3))
 temp_coeff = np.zeros((1,n_tot))
 dirr = np.zeros((n_tot,3))
 for p in range(n_phi): 
  for t in range(n_theta):
    index = n_phi * t + p  
    kappa_directional[0,index] += kappa_bulk/mfp_bulk * direction_ave[t,p]/(4.0*np.pi)*3
    temp_coeff[0,index] = domega[t,p]
 #-----------------

 #Gray 2D--------
 #n_tot = n_phi
 #kappa_directional = np.zeros((1,n_tot,3))
 #temp_coeff = np.zeros((1,n_tot))
 #dirr = np.zeros((n_tot,3))
 #for p in range(n_tot): 
 #   index = p 
 #   kappa_directional[0,p] += kappa_bulk/mfp_bulk * polar_ave[p]*Dphi/(2*np.pi)*2
 #   dirr[index] = polar_ave[p]
 #   temp_coeff[0,index] = Dphi
 #-----------------




 tc = temp_coeff/np.sum(temp_coeff)

 #Final----
 return {'tc':tc,\
         'sigma':kappa_directional[:,:,:2],\
         'kappa':kappa,\
         'mfp_average':np.array([mfp_bulk]),\
         'VMFP':dirr[:,:2],\
         'mfp_sampled':np.array([mfp_bulk]),\
         'model':np.array([1]),\
         'suppression':np.zeros(1),\
         'phi': phi  ,\
         'directions': polar  ,\
         'kappam':np.array([kappa_bulk])}


