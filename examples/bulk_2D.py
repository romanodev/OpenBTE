from openbte import Material,Geometry,Solver,Plot

Material(model='nongray',matfile='Si-300K.dat')

Geometry(model='bulk',lx=10,ly=10,step=1)

Solver(max_bte_iter=2)
#Plot(variable='vtk',repeat_x=3,repeat_y=3)


