from __future__ import print_function
import numpy as np
from scipy import interpolate


def generate_correlated_pores(**argv):
 
 #---------------READ Parameters-------
 N = argv['n']
 L = argv['l']
 d = L/N
 p = L
 le = argv['length']
 phi = argv['porosity']
 #------------------------------------
 #NOTE: Variance is 1 because we only take the smallest number up to a certain point
 gen = [np.exp(-2/le/le * np.sin(d*np.pi*i/p) ** 2) for i in range(N)]
 kernel = np.array([[  gen[abs(int(s/N) - int(t/N))]*gen[abs(s%N-t%N)]  for s in range(N*N)] for t in range(N*N)])
 y = np.random.multivariate_normal(np.zeros(N*N), kernel)
 h = y.argsort()
 idxl = h[0:int(N*N*(1-phi))]


 return idxl






  
