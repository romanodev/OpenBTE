from openbte.material import *
from openbte.geometry import *
from openbte.solver import *
from openbte.plot import *


#mat = Material(model='nongray',matfile='Si-300K.dat',n_theta=12,n_phi=48,n_mfp=10)
 
#quit()
#geo = Geometry(type='porous/square_lattice',lx=10,ly=10,lz=2,\
#                    porosity=0.25,step=1.0,shape='square')

sol = Solver(max_bte_iter = 3,max_bte_error=1e-3)


