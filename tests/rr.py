
from openbte import Material,Geometry,Solver,Plot
from openbte.utils import *

mat = Material(source='database',filename='Si',temperature=300,model='rta3D',save=False,n_theta=12,n_mfp=30,n_phi=24)
geo = Geometry(model='lattice',lx = 10,ly = 10, lz=2,step = 4, base = [[0,0]],porosity=0.2,shape='square',save=False,delete_gmsh_files=True)
sol = Solver(geometry = geo,material=mat,save=False,max_bte_iter=2)


#mat = Material(source='database',filename='Si',temperature=300,model='rta2DSym',save=False)
#geo = Geometry(model='lattice',lx = 10,ly = 10, step = 1, base = [[0,0]],porosity=0.2,shape='square',save=False,delete_gmsh_files=True)
#sol = Solver(geometry = geo,material=mat,save=False)

#print(sol['kappa'])

#print(load_data('solver_rta2DSym')['kappa'])



#sol = Solver(geometry = geo,material=mat,save=False)
#geo = Geometry(model='lattice',lx = 10,ly = 10, step = 2, base = [[0,0]],porosity=0.2,shape='square',save=False,delete_gmsh_files=True)
#sol = Solver(geometry = geo,material=mat,save=False)
