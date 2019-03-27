from openbte import Material,Geometry,Solver,Plot


Material(model='nongray',matfile='Si-300K.dat')

Geometry(porosity=0.10,lx=10,ly=10,step=1,shape='circle')

Solver()
Plot(variable='vtk',repeat_x=3,repeat_y=3)


