from openbte.geometry import *
from openbte.material import *
from openbte.solver import *
import deepdish as dd



mat = Material(model='gray',n_theta = 16, n_phi = 48,mfp=[10])
N = 1
for n in range(N):
 print(n)

 L = 10

 geo = Geometry(type='porous/random',
              lx = L,ly = L,
              step = L/10,
              Np = 10,
              phi_mean =0.3,
              automatic_periodic = False,
              save_configuration=True,
              save_fig = True)
 #------------------------------------------------
 Solver(max_bte_iter = 20,multiscale=False)
 kappa = dd.io.load('solver.hdf5')['kappa'][-1]


