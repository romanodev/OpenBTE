import numpy as np
import os

dirs = os.listdir('confs')
n_max = -1
for d in dirs:
 n_max = max([n_max,int(d.split('_')[1].split('.')[0])])

print(n_max)

