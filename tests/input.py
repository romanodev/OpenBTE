from openbte import Material,Geometry,Solver,Plot

Material(model='gray2D',mfp=1e-7,kappa=130) #mfp in nm

Geometry(model='lattice',lx = 10,ly = 10, lz=0,step = 0.5, base = [[0,0]],porosity=0.2,shape='square',delete_gmsh_files=True,direction='x')

Solver(multiscale=False,max_bte_iter=3)

#Plot(model='maps',write_html=True)
