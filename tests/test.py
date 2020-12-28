from openbte import Material,Geometry,Solver,Plot
from openbte.utils import *
from mpi4py import MPI

comm = MPI.COMM_WORLD

def rta2DSym():

    mat = Material(source='database',filename='Si',temperature=300,model='rta2DSym',save=False)
    geo = Geometry(model='lattice',lx = 10,ly = 10, step = 1, base = [[0,0]],porosity=0.2,shape='square',save=False,delete_gmsh_files=True)
    sol = Solver(geometry = geo,material=mat,save=False)

    if comm.rank == 0:
     return sol['kappa']


def test_function():

    data = rta2DSym()

    if comm.rank == 0:
     assert np.allclose(data,load_data('solver')['kappa'])


if __name__ == '__main__':

    rta2DSym()
