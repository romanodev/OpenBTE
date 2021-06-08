import h5py
import numpy as np
import scipy.linalg
from scipy.linalg import pinvh
import os
import hdfdict
import time
import h5py as h5
from .utils import *


def check_symmetric(a, rtol=1e-05, atol=1e-6):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def main():

 #Get unitcell volume
 if os.path.isfile(os.sys.argv[1]) :

  cp = []
  with open(os.sys.argv[1],'r') as fh :
    dummy  = fh.readline()
    alat   = float(fh.readline())
    for i in range(3): 
     g = fh.readline().split()
     cp.append([float(g[0]),float(g[1]),float(g[2])])  
    factor = 1
    cp = np.array(cp)*alat
    V = abs(np.dot(cp[0],np.cross(cp[1],cp[2])))*1e-30

 else:
    print('No unitcell found')
    quit()

 factor = 1

 nx = int(os.sys.argv[2])
 ny = int(os.sys.argv[3])
 nz = int(os.sys.argv[4])
 T = int(os.sys.argv[5])

 cp = np.array(cp)*alat
 V = abs(np.dot(cp[0],np.cross(cp[1],cp[2])))*1e-30

 tail = str(nx) + str(ny) + str(nz) + '.hdf5'
 
 #KAPPA-------------------------
 #f = dd.io.load('kappa-m' + tail)
 
 f = hdfdict.load('kappa-m' + tail)
 
 mode_kappa = f['mode_kappa']
 #weight = f['weight'][:]
 g = f['gamma'][:]
 gg = np.pi * g[0]*1e12 #NOTE: there should be a factor 4 here according to doc.
 (nq,nb) = np.shape(g[0])
 nm = nq*nb
 alpha = V*nq
 v = np.array(f['group_velocity'])*1e2 #m/2
 w = np.array(f['frequency'])*1e12 #1/s
 q = 1.60218e-19 #C
 kb = 1.380641e-23 #j/K
 h = 6.626070151e-34 #Js
 eta = w*h/T/kb/2
 C = kb*np.power(eta,2)*np.power(np.sinh(eta),-2) #J/K


 f = gg.reshape(nm)
 I = np.where(f > 0.0)
 tau = np.zeros(nm)
 tau[I] = 1/f[I]
 w = w.reshape(nm)
 v = np.array([v[:,:,0].reshape(nb*nq),v[:,:,1].reshape(nb*nq),v[:,:,2].reshape(nb*nq)])
 v = v.T
 C = C.reshape(nm)
 ftol = 1e-30
 index = (np.logical_and(C>ftol,f>ftol)).nonzero()[0]
 exclude = (np.logical_or(C<=ftol,f<=ftol)).nonzero()[0]

 C = C[index]
 v = v[index]
 w = w[index]
 tau = tau[index]
 sigma = np.einsum('i,ij->ij',C,v)
 kappa = np.einsum('li,lj,l,l->ij',v,v,tau,C)/alpha
 print('KAPPA (RTA):')
 print(kappa)


 data = {'C':C/alpha,'tau':tau,'v':v,'kappa':kappa}
 save_data('rta',data)   

 #---------------------------------------------

if __name__ == '__main__':
  
  main(os.sys.argv)




