import numpy as np
import sys

data = np.loadtxt(sys.argv[1])
n_mfp = np.shape(data)[0]

#Compute dis
dis = np.zeros(n_mfp)
dis[0] = data[0,1]
for m in range(n_mfp-1):
 dis[m+1] = data[m+1,1] - data[m,1]

#--------------------

f = open('mat.dat','w+')
for m in range(n_mfp):
 f.write('1    ' )
 f.write(str(m+1)+ '   ')
 tmp = data[m,0]*1e-9
 f.write('%10e' %  tmp + '   ')
 if dis[m] > 0:
  f.write('%10e' %  dis[m]+ '   ')
 else:
  f.write('%10e' %  0.0+ '   ')

 f.write('%10e' %  data[m,1]+ '\n')
f.close()


