from openbte import Geometry,Solver,Material,Plot
import deepdish as dd
import numpy as np
Material(model='gray',n_theta = 16, n_phi = 48,mfp=[10])

geometry = Geometry(  model='porous/random', lx=200, ly=200, step=10,shape='circle',
        Np=10, porosity=0.3, manual=False,only_gmsh=True,
        save_configuration=True, load_configuration=False, store_rgb=True,
        save=True, delete_gmsh_files=False,only_geo=True)

Solver(max_bte_iter=10)
Plot(variable='vtk')


