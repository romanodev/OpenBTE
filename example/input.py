import openbte.material_new2 as material
#import openbte.solver_serial2 as solver
import openbte.solver_new2 as solver
import openbte.geometry as geometry

#material.Material(matfile='Si-300K.dat',n_mfp=100,n_phi=24,n_theta=12)

#geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=2,step=1.0)
#geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=2,step=2)

#geometry.Geometry(type='bulk',porosity=0.25,shape='square',lx=10,ly=10,lz=0,step=1.0)
#
#quit()
solver.Solver(max_bte_iter=2,multiscale=False,max_fourier_iter=5)



