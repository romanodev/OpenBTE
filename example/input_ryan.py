import openbte.material_new2 as material
import openbte.solver_new3 as solver
import openbte.geometry as geometry
import time
import numpy as np

def get_porosity(L,D):
 return D*D*np.pi/4/L/L

#material.Material(matfile='Si-300K.dat',n_mfp=100,n_phi=48,n_theta=48)
#quit()


#quit()
#geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=0,step=0.5)
#geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=5,step=2.0)

#quit()
#geometry.Geometry(type='bulk',porosity=0.25,shape='square',lx=10,ly=10,lz=5,step=2)


#geometry.Geometry(type='porous/square_lattice',porosity=get_porosity(400,280),shape='circle',lx=400,ly=400,lz=250,step=100)
geometry.Geometry(type='porous/square_lattice',porosity=get_porosity(500,250),shape='circle',lx=500,ly=500,lz=250,step=100)






#geometry.Geometry(type='porous/square_lattice',porosity=phi,shape='circle',lx=400,ly=400,lz=250,step=100)





#
#quit()
#start = time.time()
#solver.Solver(max_bte_iter=15,multiscale=False,max_fourier_iter=20,load_state=False,save_state=True,max_fourier_error=1e-3)
#print(time.time() - start)


