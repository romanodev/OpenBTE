import numpy as np
from matplotlib.pylab import *


d = np.load(open('tt.dat','rb'))



plot(range(len(d)),d)
xlabel('Iteration',fontsize=18)
ylabel('Error',fontsize=18)
yscale('log')


show()



