from material import *
from geometry import *
from solver import *
from plot import *
#import mpi4py		
#import time

#mat = Material(filename='Si-300K',grid=[30,24,24],delta=0.0,min_mfp=1e-12)
mat = Material(filename='Si-300K',grid=[30,12,12],delta=0.0,min_mfp=1e-12)


geo = Geometry(model='nanowire',
               diameter = 5,
               base = 'square',
               periodic = True,
               step = 1.0,
               length = 20)


#geo = Geometry(model='load')

#L = sqrt(2.0) * 30.0
#geo = Geometry(model='bulk',
#               frame = [100,L,L],
#               direction = 'x',
#               porosity = 0.25,
#               step = 10.0,
#               Periodic=[True,True,True],
#               shape = 'square')

 #find next n


sol = Solver(model='bte',material=mat,geometry=geo,max_bte_iter = 0,multiscale=True)

#Plot(model='map/flux_magnitude',material=mat,geometry=geo)
#Plot(plot='suppression_function',material=mat,geometry=geo)

























