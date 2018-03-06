from material import *
from geometry import *
from solver import *
from plot import *
import mpi4py		
import time



mat = Material(filename='Si-300K',grid=[50,12,48],delta=0.0,min_mfp=1e-12)


geo = Geometry(model='porous/random',
               frame = [10.0,10.0],
               porosity = 0.15,
               step =1.0,
               shape = 'square',
               Nx = 2,
               Ny = 2,
               save_configuration = False)


#geo = Geometry(model='bulk',
#               frame = [100.0,100.0],
#               porosity = 0.25,
#               step = 10.0,
#               shape = 'square')

 #find next n


sol = Solver(model='bte',material=mat,geometry=geo,max_bte_iter = 10,multiscale=False,repeat_x = 2, rrepeat_y = 2)

#Plot(model='map/flux_magnitude',material=mat,geometry=geo)
#Plot(plot='suppression_function',material=mat,geometry=geo)

























