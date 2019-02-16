from openbte import Material,Geometry,Solver,Plot
Material(matfile ='Si-300K.dat')
Geometry(porosity=0.25,lx=10,ly=10,step=1.0)
Solver()
Plot(variable='vtk')


