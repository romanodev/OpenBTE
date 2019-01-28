import openbte.material_new2 as material
import openbte.solver_new3 as solver
import openbte.geometry as geometry
import time

#material.Material(matfile='Si-300K.dat',n_mfp=100,n_phi=48,n_theta=48)
#quit()


#quit()
#geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=2,step=1)
#geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=5,step=2.0)

#quit()


#
#quit()
#start = time.time()
solver.Solver(max_bte_iter=10,multiscale=False,max_fourier_iter=20,load_state=False,save_state=True,max_fourier_error=1e-3)
#print(time.time() - start)


