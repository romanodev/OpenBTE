from openbte import Material,Geometry,Solver,Plot

Material(model='nongray',matfile='Si-300K.dat')
Geometry(porosity=0.30,lx=10,ly=10,lz=2,step=2,shape='square')
Solver(max_bte_iter=10)
Plot(variable='variable/flux')



