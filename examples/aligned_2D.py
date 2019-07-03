from openbte import Material,Geometry,Solver,Plot

import openbte


Material(model='nongray',matfile='Si-300K.dat')
Geometry(porosity=0.30,lx=10,ly=10,step=1,shape='square')
data = Solver(max_bte_iter=0)
Plot(variable='variable/flux')



