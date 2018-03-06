import numpy as np
import os
import deepdish as dee
from matplotlib.pylab import *

#dirs = os.listdir('confs')
#kappa = []#
#confs = []
#for dd in dirs:
# if dd.split('_')[0] == 'kappa':
#  a = float(np.load('confs/' + dd))
#  if a > 0.0: 
#    conf = np.load('confs/conf_' + str(dd.split('_')[1]))
#    confs.append(conf.flatten()/20.0)
#    kappa.append(a)
#confs = np.array(confs)
#kappa = np.array(kappa)

#kappa.dump(open('kappa_4.dat','wb'))
#confs.dump(open('confs_4.dat','wb'))

#quit()


#print(np.shape(kappa))

#quit()

#dee.io.save('data.hdf5',{'X':confs,'y':kappa})

#quit()


data = dee.io.load('data2.hdf5')
confs = data['X']
kappa = data['y']


#kappa  = np.array(kappa)*100.0

print(len(kappa))
#print(np.shape(confs))

#quit()
#print(X)
#print(y)

quit()



#quit()
validation = False

if validation:

 CV = 100

 kappa = kappa[500:500 + CV]
 y = np.array([kappa]).T
 X = np.array(confs)[500:500+CV]

 ann = dee.io.load('ann.hdf5')
 nu = ann['nu']
 NL = ann['NL']
 syn = ann['syn']
 l = X
 for n in range(NL-1):
  l = 1/(1+np.exp(-(np.dot(l,syn[n]))))
 error = y-l
 print "Error:" + str(np.mean(np.abs(error)))

 plot(range(CV),y,'r',label='correct')
 plot(range(CV),l,'b',label='approximated')
 legend()
 show()
 

else:

 N = 100
 kappa = kappa[0:N]
 y = np.array([kappa]).T #/np.max(kappa)
 X = np.array(confs)[0:N]

 ni = len(X[0])
 no = len(y[0])

 nu =32
 NL = 4
 err = []
 #---------------------------------------------------
 syn = {0:2*np.random.random((ni,nu)) - 1}
 for n in range(NL-3):
  syn.update({n+1:2*np.random.random((nu,nu)) - 1})
 syn.update({NL-2:2*np.random.random((nu,no)) - 1})
#---------------------------------------------------

 l = {0:X}
 for j in xrange(10000):
   for n in range(NL-1):
    l.update({n+1:1/(1+np.exp(-(np.dot(l[n],syn[n]))))})

   error = y-l[NL-1]
   if (j% 100) == 0:
    err.append(np.mean(np.abs(error)))
    print "Error:" + str(np.mean(np.abs(error))) + 'Progress: ' + str(float(j)/10000)

   for n in range(NL-1)[::-1]:
    delta = error*(l[n+1]*(1-l[n+1]))
   
    syn[n] += 0.25*l[n].T.dot(delta)
    error = delta.dot(syn[n].T)
   

 network = {'syn':syn,'nu':nu,'NL':NL}

 dee.io.save('ann.hdf5',network)
 err = np.array(err)
 err.dump(open('tt.dat','wb'))




