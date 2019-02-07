import openbte.material as material
import openbte.solver as solver
import openbte.geometry as geometry
import time
import numpy as np

def get_porosity(L,D):
 return D*D*np.pi/4/L/L

#material.Material(matfile='TDP.dat',n_mfp=100,n_phi=48,n_theta=48)
#quit()


#geometry.Geometry(type='porous/square_lattice',porosity=get_porosity(400,280),shape='circle',lx=400,ly=400,lz=250,step=100)

#quit()
#geometry.Geometry(type='porous/square_lattice',porosity=get_porosity(500,250),shape='circle',lx=500,ly=500,lz=250,step=100)



solver.Solver(max_bte_iter=15,multiscale=True,max_fourier_iter=20,load_state=False,save_state=True,max_fourier_error=1e-3)



