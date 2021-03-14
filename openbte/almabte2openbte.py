import numpy as np
import sys
from .utils import *

def main():

 filename = sys.argv[1]
 if len(sys.argv) == 3:
    output = sys.argv[2]
 else  :
    output = 'rta'


 tmp = np.loadtxt(filename,skiprows=1,delimiter=',')
 nq = int(np.max(tmp[:,0]))+1
 nb = int(np.max(tmp[:,1]))+1
 ntot = nq*nb

 tau = tmp[:,7]
 v   = tmp[:,8:]
 C   = tmp[:,6]
 w   = tmp[:,5]
 kappa =  np.einsum('ki,kj,k,k',v,v,tau,C) 

 data = {'C':C,'tau':tau,'v':v,'kappa':kappa,'f':w/2.0/np.pi}

 save_data(output,data)   
