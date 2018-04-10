import numpy as np
import sys


def main() :

 data = np.loadtxt('BTE.cumulative_kappa_scalar')
 n_mfp = np.shape(data)[0]

 #Compute dis
 dis = np.zeros(n_mfp)
 dis[0] = data[0,1]
 for m in range(n_mfp-1):
  dis[m+1] = data[m+1,1] - data[m,1]

 #--------------------
 f = open('mat.dat','w+')
 for m in range(n_mfp):
  tmp = data[m,0]*1e-9
  f.write('%10e' %  tmp + '   ')
  f.write('%10e' %  dis[m]+ '\n')
 f.close()

 if __name__ == "__main__":
   main()
