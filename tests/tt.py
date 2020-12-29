from openbte import Material,Geometry,Solver,Plot
from openbte.utils import *
from mpi4py import MPI

comm = MPI.COMM_WORLD


# content of test_class.py
class TestClass(object):

    def test_rta2DSym(self):

      mat = Material(source='database',filename='Si',temperature=300,model='rta2DSym',save=False)
      geo = Geometry(model='lattice',lx = 10,ly = 10, step = 1, base = [[0,0]],porosity=0.2,shape='square',save=False,delete_gmsh_files=True)
      sol = Solver(geometry = geo,material=mat,save=False)
      if comm.rank == 0:
       assert np.allclose(sol['kappa'],load_data('solver_rta2DSym')['kappa'])


    def test_rta3D(self):

     mat = Material(source='database',filename='Si',temperature=300,model='rta3D',save=False,n_theta=12,n_mfp=30,n_phi=24)
     geo = Geometry(model='lattice',lx = 10,ly = 10, lz=2,step = 2, base = [[0,0]],porosity=0.2,shape='square',save=False,delete_gmsh_files=True)
     sol = Solver(geometry = geo,material=mat,save=False,max_bte_iter=10)
     if comm.rank == 0:
      assert np.allclose(sol['kappa'],load_data('solver_rta3D')['kappa'])


#def rta2DSym():

#    mat = Material(source='database',filename='Si',temperature=300,model='rta2DSym',save=False)
#    geo = Geometry(model='lattice',lx = 10,ly = 10, step = 1, base = [[0,0]],porosity=0.2,shape='square',save=False,delete_gmsh_files=True)
#    sol = Solver(geometry = geo,material=mat,save=False)
#    if comm.rank == 0:
#     assert np.allclose(sol['kappa'],load_data('solver_rta2DSym')['kappa'])


#def test_rta3D():

#    mat = Material(source='database',filename='Si',temperature=300,model='rta3D',save=False,n_theta=12,n_mfp=30,n_phi=24)
#    geo = Geometry(model='lattice',lx = 10,ly = 10, lz=2,step = 2, base = [[0,0]],porosity=0.2,shape='square',save=False,delete_gmsh_files=True)
#    sol = Solver(geometry = geo,material=mat,save=False,max_bte_iter=10)
#    if comm.rank == 0:
#     assert np.allclose(sol['kappa'],load_data('solver_rta3D')['kappa'])


#    if comm.rank == 0:
#     return sol['kappa']

#def test_function():

    #data = rta2DSym()
    #if comm.rank == 0:
    # assert np.allclose(data,load_data('solver_rta2DSym')['kappa'])

    #comm.Barrier()
#    data = rta3D()
#    if comm.rank == 0:
#     assert np.allclose(data,load_data('solver_rta3D')['kappa'])

#if __name__ == '__main__':

    #rta2DSym()
#    rta3D()
