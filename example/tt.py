import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')



ax.scatter(range(5),range(5),range(5))

plt.show()
