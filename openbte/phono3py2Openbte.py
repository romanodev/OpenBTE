import h5py
import numpy as np
import scipy.linalg
from numpy.linalg import pinv
import os
import deepdish as dd

def check_symmetric(a, rtol=1e-05, atol=1e-6):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def main(argv):

 #Get unitcell volume
 if os.path.isfile(argv[1]) :

  cp = []
  with open(argv[1],'r') as fh :
    dummy  = fh.readline()
    alat   = float(fh.readline())
    for i in range(3): 
     g = fh.readline().split()
     cp.append([float(g[0]),float(g[1]),float(g[2])])  
    factor = 1
    cp = np.array(cp)*alat
    V = abs(np.dot(cp[0],np.cross(cp[1],cp[2])))*1e-30

 else: print('No unitcell found')

 factor = 1

 nx = int(argv[2])
 ny = int(argv[3])
 nz = int(argv[4])

 cp = np.array(cp)*alat
 V = abs(np.dot(cp[0],np.cross(cp[1],cp[2])))*1e-30

 tail = str(nx) + str(ny) + str(nz) + '.hdf5'
 f = dd.io.load('unitary-m' + tail)
 Q = f['unitary_matrix'][0,0]
 nq,nb,_,_ = np.shape(Q)
 nm = nb * nq
 f = dd.io.load('coleigs-m' + tail)
 D = f['collision_eigenvalues'][0]
 D = np.diag(D)

 Q = Q.reshape(nm,nm)
 O = np.matmul(Q.T,np.matmul(D,Q))

 f = dd.io.load('kappa-m' + tail)
 mode_kappa = f['mode_kappa']
 weight = f['weight'][:]
 g = f['gamma'][:]
 gg = 2 * 2 * np.pi * g[0]*1e12
 (nq,nb) = np.shape(g[0])
 nm = nq*nb
 v = np.array(f['group_velocity'])*1e2 #m/2
 w = np.array(f['frequency'])*1e12 #1/s

 C = np.array(f['heat_capacity'])[0]*1.60218e-19/V/nq*factor 

 #tau = tau.reshape(nm)
 w = w.reshape(nm)
 v = np.array([v[:,:,0].reshape(nb*nq),v[:,:,1].reshape(nb*nq),v[:,:,2].reshape(nb*nq)])
 v = v.T
 C = C.reshape(nm)
 f = gg.reshape(nm)
 A = O/(1e-12/np.pi)

 #compute inverse of sqrt(C)--------------------------
 I = np.where(C > 0.0)
 invSqrtC = np.zeros(nm)
 invSqrtC[I] = 1/np.sqrt(C[I])
 #------------------------------
 I = np.where(f > 0.0)
 tau= np.zeros(nm)
 tau[I] = 1/f[I]
 #--------------
 I = np.where(C > 0.0)
 invC = np.zeros(nm)
 invC[I] = 1/C[I]
 #-----------------------------
 sigma = np.einsum('i,ij->ij',C,v)

 Omega = np.einsum('ij,i,j->ij',A,invSqrtC,np.sqrt(C))
 W = np.einsum('lk,l->lk',Omega,C)

 #W = 0.5*W + 0.5*W.T

 kappa = np.einsum('li,lk,kj->ij',sigma,pinv(W),sigma)


 data = {'W':W,'v':v,'C':C,'kappa':kappa}
 dd.io.save('full.h5',data)

if __name__ == '__main__':

    main(os.sys.argv)




