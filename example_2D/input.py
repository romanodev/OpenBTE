import openbte.material_new2 as material
import openbte.solver_new3 as solver
import openbte.geometry as geometry
import time

#material.Material(matfile='Si-300K.dat',n_mfp=100,n_phi=48,n_theta=48)
#geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=0,step=1.0)
#quit()
solver.Solver(max_bte_iter=20,multiscale=True,max_fourier_iter=20,load_state=False,save_state=True,max_fourier_error=1e-8,fourier_cut_mfp = 100)


