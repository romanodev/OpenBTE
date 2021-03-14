import sys
import numpy as np
from .utils import *


def gray2D(**argv):

 #Import data----
 kappa_bulk = argv['kappa']
 kappa = argv['kappa']*np.eye(2)
 mfp_bulk = argv['mfp']
 n_phi = int(argv.setdefault('n_phi',48))
 Dphi = 2*np.pi/n_phi
 phi = np.linspace(Dphi/2.0,2.0*np.pi-Dphi/2.0,n_phi,endpoint=True)
 polar = np.array([np.sin(phi),np.cos(phi)]).T
 fphi= np.sinc(Dphi/2.0/np.pi)
 polar_ave = polar*fphi
 #------------------

 kappa_directional = np.zeros((1,n_phi,2))
 temp_coeff = np.zeros((1,n_phi))
 temp_coeff[0,:] = 1/n_phi
 suppression = np.zeros((1,n_phi,1)) 
 for p in range(n_phi): 
    kappa_directional[0,p] += kappa_bulk/mfp_bulk * polar_ave[p]/n_phi*2

 tc = temp_coeff/np.sum(temp_coeff)


 #Final----
 return {'tc':tc,\
         'sigma':kappa_directional,\
         'kappa':kappa,\
         'mfp_average':np.array([mfp_bulk]),\
         'VMFP':polar_ave,\
         'mfp_sampled':np.array([mfp_bulk]),\
         'model':np.array([1]),\
         'suppression':np.zeros(1),\
         'phi': phi  ,\
         'directions': polar  ,\
         'kappam':np.array([kappa_bulk])}


