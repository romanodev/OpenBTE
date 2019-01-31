import openbte.material_new2 as material
import openbte.solver_new3 as solver
import openbte.geometry as geometry
import openbte.plot as plot
import time
#
#material.Material(matfile='Si-300K.dat',n_mfp=40,n_phi=24,n_theta=12)
#quit()


#quit()
#geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=0,step=1)
#geometry.Geometry(type='bulk',porosity=0.25,shape='square',lx=10,ly=10,lz=0,step=1)
#geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=5,step=2.0)

#quit()
#geometry.Geometry(type='bulk',porosity=0.25,shape='square',lx=10,ly=10,lz=5,step=2)

#
#quit()
#start = time.time()
solver.Solver(max_bte_iter=10,multiscale=False,max_fourier_iter=20,load_state=False,save_state=True,max_fourier_error=1e-3)
#print(time.time() - start)

#plot.Plot(variable='vtk')
