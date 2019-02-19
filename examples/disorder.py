from openbte import Geometry,Solver,Material,Plot
import deepdish as dd

#Material(model='gray',n_theta = 16, n_phi = 48,mfp=[10])
#L = 10
#Geometry(model='porous/random',
#              lx = L,ly = L,lz=0,
#              step = L/5,
#              shape='square',
#              Np = 10,
#              phi_mean = 0.3,
#              manual = False,
#              centers = [[0.5,0.5]],
#              automatic_periodic = False,
#              save_configuration=True,
#              save_fig = True)
#------------------------------------------------
#Solver(max_bte_iter = 20,multiscale=False)
#kappa = dd.io.load('solver.hdf5')['kappa'][-1]
Plot(variable='map')
#Plot(variable='vtk',repeat_x = 2, repeat_y = 2)

