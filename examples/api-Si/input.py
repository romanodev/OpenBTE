from material import *
from geometry import *
from solver import *
from plot import *
#import mpi4py		
#import time

#mat = Material(filename='Si-300K',grid=[30,12,48],delta=0.0,min_mfp=1e-12)

geo = Geometry(model='porous/custom',
              frame = [10.0,10.0,3.0],
              polygons = [[[-0.5,-0.5],[-0.5,0.5],[0.5,0.5],[0.5,-0.5]]],
              #porosity = 0.25,
              step = 1.0)
              #shape = 'square')

#geo = Geometry(model='porous/aligned',
#              frame = [10.0,10.0],
#              porosity = 0.025,
#              step = 1.0,
#              shape = 'square')

#geo = Geometry(model='load')
#mat = Material(model='load')
#geo = Geometry(model='bulk',
#               frame = [10.0,10.0],
#               reservoirs = False,
#               unstructured = False,
#               step = 0.5)
#if MPI.COMM_WORLD.Get_rank() == 0:
# a = time.time()
#sol = Solver(max_bte_iter = 5,multiscale=True,repeat_x = 1, repeat_y = 1)
#if MPI.COMM_WORLD.Get_rank() == 0:
# print(time.time() - a)

#Plot(model='map/flux_magnitude',material=mat,geometry=geo)

#Plot(model='suppression_function/gg')

























