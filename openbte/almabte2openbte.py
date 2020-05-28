import deepdish
import numpy as np
from numpy import inf
import sys
import deepdish as dd

def main():

 #constants
 kb = 1.380649e-23 #J/K
 hbar = 1.054571817e-34#J*s/rad
 #---------

 filename = sys.argv[1]

 T = float(filename.split('.')[0].split('_')[-1].split('K')[0])


 tmp = np.loadtxt(filename,skiprows=1,delimiter=',')

 nq = int(np.max(tmp[:,0]))+1
 nb = int(np.max(tmp[:,1]))+1
 ntot = nq*nb

 tau = np.zeros(ntot)
 C = np.zeros(ntot)
 w = np.zeros(ntot)
 v = np.zeros((ntot,3))
 for n in range(ntot):
  index = int(nq * tmp[n,1] + tmp[n,0])
  tau[index] = tmp[n,7]
  C[index] = tmp[n,6]  
  v[index,0] = tmp[n,8]  
  v[index,1] = tmp[n,9]  
  v[index,2] = tmp[n,10]  
  w[index] = tmp[n,5]  

 eta = w*hbar/kb/T/2

 tmp = np.sinh(eta)
 tmp[tmp < 1e-20] = 1e40

 C2 = kb*np.power(eta/tmp,2)
 C2 = np.nan_to_num(C2)
 g = np.nan_to_num(C2)/C

 alpha = g[g.nonzero()[0][0]]

 kappa = np.zeros((3,3))
 for n in range(ntot):
    kappa += C2[n]*tau[n]*np.outer(v[n],v[n])

 tau[tau < 1e-20] = 1e40

 data = {'alpha':alpha,'f':w/2/np.pi,'tauinv':1/tau,'v':v,'kappa':kappa}

 
 dd.io.save('rta.h5',data)


if __name__ == '__main__':

    main()







