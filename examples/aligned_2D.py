from openbte import Material,Geometry,Solver,Plot


Material(model='gray',mfp=[10],kappa_bulk=[10])

#Geometry(porosity=0.25,lx=10,ly=10,step=1,inclusion=False)
Solver()
#Plot(variable='map/flux',direction='magnitude')
#Plot(variable='map')
#Plot(variable='SUP')
#Plot(variable='vtk')


