import deepdish
import numpy as np
from numpy import inf
import sys
import deepdish as dd

def main():

 filename = sys.argv[1]
 tmp = np.loadtxt(filename,skiprows=1,delimiter=',')
 nq = int(np.max(tmp[:,0]))+1
 nb = int(np.max(tmp[:,1]))+1
 ntot = nq*nb

 tau = tmp[:,7]
 v   = tmp[:,8:10]
 C   = tmp[:,6]

 kappa =  np.einsum('ki,kj,k,k',v,v,tau,C) 

 data = {'C':C,'tau':tau,'v':v,'kappa':kappa}
 dd.io.save('rta.h5',data)
 
