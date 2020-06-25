import deepdish
import numpy as np
from numpy import inf
import sys
import deepdish as dd


def main():

 data = dd.io.load('rta.h5')
 kappam = np.einsum('u,u,u,u->u',data['tau'],data['C'],data['v'][:,0],data['v'][:,0]) 
 mfp_bulk = np.einsum('ki,k->ki',data['v'],data['tau'])

 mfp = np.array([np.linalg.norm(m) for m in mfp_bulk])

 I = np.argsort(mfp)

 kappa = kappam[I]
 mfp = mfp[I]

 dd.io.save('mfp.h5',{'K':kappa,'mfp':mfp})




 
