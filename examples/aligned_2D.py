from openbte import Material,Geometry,Solver,Plot

Material(model='nongray',matfile='Si-300K.dat')
Geometry(porosity=0.30,lx=10,ly=10,step=1,shape='square')
Solver(max_bte_iter=2,write_pseudotemperature=True)
#Plot(variable='variable/flux')
#Plot(variable='vtk')



