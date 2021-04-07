import numpy as np
from numpy import inf
import sys
from .utils import *
import sys


def main():
 data = load_data(sys.argv[1][:-4])

 kappam = np.einsum('u,u,u,u->u',data['tau'],data['C'],data['v'][:,0],data['v'][:,0]) 
 mfp_bulk = np.einsum('ki,k->ki',data['v'],data['tau'])

 mfp = np.array([np.linalg.norm(m) for m in mfp_bulk])

 I = np.argsort(mfp)

 kappa = kappam[I]
 mfp = mfp[I]
 Kacc = np.zeros_like(kappa)
 
 for n,k in enumerate(kappa):
     if n == 0: 
        Kacc[n] = k
     else:  
        Kacc[n] = k + Kacc[n-1]

 #select only meaningful MFPs
 a = np.where(Kacc < Kacc[-1]*1e-5)[0][-1]
 Kacc = Kacc[a:]
 mfp =  mfp[a:]

 #DownSamplimg
 N = 100

 mfp_sampled = np.logspace(np.log10(min(mfp)),np.log10(max(mfp)),N,endpoint=True)
 mfp_sampled[-1]= mfp[-1]
 
 
 Kacc_d = np.zeros(N)
 for n,m in enumerate(mfp_sampled):
    Kacc_d[n] = Kacc[np.where(mfp >= m)[0][0]]

 Kacc_d[-1] = Kacc[-1]
 save_data('mfp',{'Kacc':np.array(Kacc_d),'mfp':np.array(mfp_sampled)})




 
