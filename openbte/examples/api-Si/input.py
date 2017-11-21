from openbte.material import *
from openbte.geometry import *
from openbte.solver import *
from openbte.plot import *

mat = Material(filename='Si-300K',grid=[25,6,32])

geo = Geometry(model='porous/aligned',
               frame = [10.0,10.0],
               porosity = 0.25,
               step = 1.0,
               shape = 'square')

sol = Solver(model='bte',material=mat,geometry=geo,max_bte_iter=10)

Plot(model='map/flux_magnitude',material=mat,geometry=geo)
























