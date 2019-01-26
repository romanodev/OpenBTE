import openbte.material_new2 as material
import openbte.solver_new3 as solver
import openbte.geometry as geometry
import time

#material.Material(matfile='Si-300K.dat',n_mfp=30,n_phi=48,n_theta=24)

#geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=0,step=0.5)
#geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=2,step=0.75)

#quit()
#geometry.Geometry(type='bulk',porosity=0.25,shape='square',lx=10,ly=10,lz=0,step=0.75)
#
#quit()
#start = time.time()
solver.Solver(max_bte_iter=10,multiscale=False,max_fourier_iter=20,load_state=False,save_state=True)
#print(time.time() - start)


