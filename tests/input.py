from openbte import Material,Geometry,Solver,Plot

mat = Material(source='database',filename='rta_Si_300',model='rta3D',save=False,n_theta=12,n_mfp=30,n_phi=24)
geo = Geometry(model='lattice',lx = 10,ly = 10, lz=2,step = 2, base = [[0,0]],porosity=0.2,shape='square',save=False,delete_gmsh_files=True)
sol = Solver(geometry = geo,material=mat,save=True,max_bte_iter=10)


