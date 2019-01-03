import openbte

#print(openbte.__file__)

#quit()

from openbte.geometry import *
from openbte.solver_serial2 import *
from openbte.material_new import *
#from openbte.material import *
#from openbte.plot import *


#mfps = np.logspace(-12,-4,30)
#mat = Material(model='gray',n_theta = 48, n_phi = 96,mfps=mfps,special_point=-1,polar_offset = 0)
##mat = Material(model='gray',n_theta = 48, n_phi = 48,mfps=mfps,polar_offset = 0)
mat = Material(model='nongray',matfile='Si-300K.dat',n_phi=24,n_rmfp=100,polar_offset=7.5)
#mat = Material(model='nongray',matfile='Si-300K.dat',n_theta=24,n_phi=24,n_mfp=50,polar_offset=7.5)

 
#quit()
#geo = Geometry(type='bulk',lx=10,ly=10,lz=2,\
#                    porosity=1.0,step=1.0,\
#                    shape='square')



#geo = Geometry(type='porous/square_lattice',lx=10,ly=10,lz=2,porosity=0.25,step=1.0,shape='square')

sol = Solver(max_bte_iter=5,max_bte_error=1e-6,max_fourier_iter=20,multiscale=False)


