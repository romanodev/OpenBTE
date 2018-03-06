import numpy as np
import os
from matplotlib.pylab import *

dirs = os.listdir('confs')
kappa = []
confs = []

for dd in dirs:
 if dd.split('_')[0] == 'kappa':
  a = float(np.load('confs/' + dd))
  kappa.append(a)
 else:
  a = np.load('confs/' + dd)
  confs.append(a)

confs = np.array(confs)

plot(kappa)
show()













