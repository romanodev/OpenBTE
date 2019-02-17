from openbte import Material,Solver,Geometry


mat = Material(model='gray',n_theta = 16, n_phi = 48,mfp=[10])

L = 10
N = 10
NP= 10

geo = Geometry(model='porous/random_over_grid',
              lx = L,ly = L,
              nx = N, ny = N,np=NP,
              step = L/10,
              automatic_periodic = False)

x = geo.x
#------------------------------------------------

Solver(max_bte_iter = 10,multiscale=False)

kappa  = dd.io.load('solver.hdf5')['kappa'][-1]



