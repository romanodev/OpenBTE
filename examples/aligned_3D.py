from openbte import Material,Geometry,Solver,Plot

#Material(model = 'gray',mfp = [100])
Geometry(porosity=0.25,lx=10,ly=10,lz=2,step=2,only_gmsh=True)
#L = 10
#Geometry(model='porous/random',
#              lx = L,ly = L,lz=1,
#              step = L/5,
#              shape='circle',
#              Np = 10,
#              phi_mean = 0.3,
#              only_geo = True,
#              manual = False,
#              centers = [[0.0,0.0]],
#              automatic_periodic = False,
#              save_configuration=True,
#              save_fig = True,
#              only_gmsh=True)

#quit()
#------------------------------------------------

#Solver(max_bte_iter=10)
#Plot(variable='map')
#Plot(variable='vtk')


