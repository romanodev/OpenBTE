from material import *
from geometry import *
from solver import *
from plot import *
import mpi4py		
import time

mat = Material(filename='Si-300K',grid=[50,12,48],delta=0.0,min_mfp=1e-12)

geo = Geometry(model='porous/aligned',
              frame = [10.0,10.0],
              porosity = 0.25,
              step = 0.5,
              shape = 'square')


#geo = Geometry(model='load')


#geo = Geometry(model='bulk',
#               frame = [10.0,10.0],
#               reservoirs = False,
#               unstructured = False,
#               step = 0.5)

#if MPI.COMM_WORLD.Get_rank() == 0:
# a = time.time()
sol = Solver(material=mat,geometry=geo,max_bte_iter = 0,multiscale=True,repeat_x = 2, repeat_y = 2)
#if MPI.COMM_WORLD.Get_rank() == 0:
# print(time.time() - a)

#Plot(model='map/flux_magnitude',material=mat,geometry=geo)
#Plot(plot='suppression_function',material=mat,geometry=geo)

























