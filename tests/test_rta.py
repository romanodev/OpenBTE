from openbte import Material,Geometry,Solver,Plot
from openbte.utils import *
from mpi4py import MPI
comm = MPI.COMM_WORLD

mat = Material(source='database',filename='rta_Si_300',model='rta2DSym',save=False)

class TestClass(object):

    def test_rta2DSym(self):

      geo = Geometry(model='lattice',lx = 10,ly = 10, step = 1, base = [[0,0]],porosity=0.2,shape='square',save=False,delete_gmsh_files=True)
      sol = Solver(geometry = geo,material=mat,save=False)

      if comm.rank == 0:
       assert np.allclose(sol['kappa'],load('solver_2DSym')['kappa'],rtol=1e-3)


    def test_disk(self):

       geo = Geometry(model='disk',Rh=1,R=10,step=1,heat_source=0.5,save=False)

       sol = Solver(geometry = geo,material=mat,save=False)

       if comm.rank == 0:
         assert np.allclose(sol['Temperature_BTE'],load('solver_disk')['Temperature_BTE'],rtol=1e-3)








