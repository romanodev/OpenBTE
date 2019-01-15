import openbte.material_new2 as material
#import openbte.solver_serial2 as solver
import openbte.solver_new2 as solver
import openbte.geometry as geometry

#material.Material(matfile='Si-300K.dat',n_mfp=30,n_phi=48,n_theta=24)

#geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=0,step=1.0)
#geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=2,step=0.75)


#geometry.Geometry(type='bulk',porosity=0.25,shape='square',lx=10,ly=10,lz=0,step=0.75)
#
#quit()
solver.Solver(max_bte_iter=10,multiscale=False,max_fourier_iter=20,load_state=True,save_state=True)



