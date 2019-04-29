from openbte import Geometry,Solver,Material,Plot
import deepdish as dd
import numpy as np
#Material(model='gray',n_theta = 16, n_phi = 48,mfp=[10])
#L = 10

geometry = Geometry(
        model='porous/random', lx=10, ly=10, step=1,
        Np=10, porosity=0.3, manual=False,
        save_configuration=False, load_configuration=True, store_rgb=True,
        save=False, delete_gmsh_files=False)

#Geometry(model='porous/random',
#              lx = L,ly = L,lz=0,
#              step = L/5,
#              shape='circle',
#              Np = 10,
#              phi_mean = 0.3,
#              manual = True,
#              load_configuration=True,
#              automatic_periodic = False,
#              save_configuration=True,
#              save_fig = False)
#------------------------------------------------
#Solver(max_bte_iter=10)
#Plot(variable='vtk')


