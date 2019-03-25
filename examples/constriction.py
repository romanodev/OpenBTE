from openbte import Material,Geometry,Solver,Plot



Material(model='nongray',matfile='Si-300K.dat')
Geometry(model='porous/custom',polygons=[0.45,0.3,0.55,0.3,0.55,-0.3,0.45,-0.3],lx=500,ly=500,lz=0,step=25,only_gmsh=False)
Solver(multiscale=True)

Plot(variable='vtk',repeat_x=3,repeat_y=3)


