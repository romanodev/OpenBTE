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

 #---------------------------------

 #FULL MATRIX----------------------------------
 fname = 'unitary-m' + tail
 #f = dd.io.load(fname)

 f = hdfdict.load('unitary-m' + tail)
 Q = f['unitary_matrix'][0,0]
  
 #f = dd.io.load('coleigs-m' + tail)
 f = hdfdict.load('coleigs-m' + tail)
 D = f['collision_eigenvalues'][0]
 Dm = np.diag(D)
 Q = Q.reshape(nm,nm)
 
 

 #A = np.matmul(Q.T,np.matmul(Dm,Q))*1e12*np.pi
 
 QT = Q.T
 A = np.matmul(QT,(D*QT).T)
 A = np.delete(A,exclude,0)
 A = np.delete(A,exclude,1)


 #a = np.einsum('ij,i->j',A,np.sqrt(C))
 #print(sum(a))
 #print(sum(np.absolute(a)))
 #show()

 W = np.einsum('ij,i,j->ij',A,np.sqrt(C),np.sqrt(C))*1e12*np.pi
 #W = np.einsum('ij,i,j->ij',A,1/np.sqrt(C),1/np.sqrt(C))*1e12*np.pi

 print('Start inversion...')
 kappa = np.einsum('li,lk,kj->ij',sigma,pinvh(W),sigma)/alpha
 print('KAPPA (FULL):')
 print(kappa)

 data = {'W':W,'v':v,'C':C,'kappa':kappa,'alpha':np.array([alpha])}
 #np.savez_compressed('full.npz',data)   
 #Saving data
 save_data('full',data)   
 #hdfdict.dump(data,'full.h5')

 data = {'C':C/alpha,'tau':tau,'v':v,'kappa':kappa}
 saveCompressed('rta.npz',**data)   
 #save_data('full',data)   
 #np.savez_compressed('rta.npz',data)   

 #---------------------------------------------

if __name__ == '__main__':
  
  main(os.sys.argv)




