from openbte import Material,Geometry,Solver


mat = Material(model='gray', n_theta=16, n_phi=48, mfp=[10],save=False)

geo = Geometry(model='porous/random', lx=10, ly=10, step=1,
                Np=10, porosity=0.1, manual=False,save_configuration=False,store_rgb=True,save=False,delete_gmsh_files=True)

sol = Solver(material=mat,geometry=geo, max_bte_iter=20, multiscale=False,verbose=False,save=False,save_data=False,keep_lu=True)

#output-----
kappa = sol.state['kappa'][-1]
rgb = geo.rgb
#----------------



