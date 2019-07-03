from openbte import Material,Solver,Geometry


#mat = Material(model='gray',n_theta = 16, n_phi = 48,mfp=[10])
L = 10
N = 10
NP= 20

geo = Geometry(model='porous/random_over_grid',inclusion=False,
              lx = L,ly = L,lz=0,
              nx = N, ny = N,np=NP,
              save_configuration=True,
              step = L/10,
              automatic_periodic = False)
#------------------------------------------------

#(cx,cy) = geo.get_elem_centroid(0)

#data = Solver(max_bte_iter = 10,multiscale=False).state

#print(data['flux'])




