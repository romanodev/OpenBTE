
from openbte import Material,Geometry,Solver,Plot

mat = Material(source='database',filename='Si',temperature=300,model='rta2DSym',save=False)
geo = Geometry(model='lattice',lx = 10,ly = 10, step = 1, base = [[0,0]],porosity=0.2,shape='square',save=False,delete_gmsh_files=True)
sol = Solver(geometry = geo,material=mat,save=False)
geo = Geometry(model='lattice',lx = 10,ly = 10, step = 2, base = [[0,0]],porosity=0.2,shape='square',save=False,delete_gmsh_files=True)
sol = Solver(geometry = geo,material=mat,save=False)
