
import deepdish as dd
import numpy as np


tmp = dd.io.save('rr',{'d':np.arange(5),'f':np.arange(6)})
tmp = dd.io.load('rr')
