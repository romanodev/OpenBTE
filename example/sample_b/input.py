import openbte.material as material
import openbte.solver as solver
import openbte.geometry as geometry
import time
import numpy as np
import openbte.plot as plot

def get_porosity(L,D):
 return D*D*np.pi/4/L/L

material.Material(matfile='TDP.dat',n_mfp=100,n_phi=48,n_theta=48)



geometry.Geometry(type='porous/square_lattice',porosity=get_porosity(500,250),shape='circle',lx=500,ly=500,lz=250,step=30)



solver.Solver(max_bte_iter=30,multiscale=True,max_fourier_iter=20,load_state=False,save_state=True,max_fourier_error=1e-3)

#plot.Plot(variable='SUP')


