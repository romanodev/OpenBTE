import openbte.material as material
import openbte.solver as solver
import openbte.geometry as geometry
import openbte.plot as plot


material.Material(matfile='Si-300K.dat',n_mfp=40,n_phi=24,n_theta=12)

geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=0,step=0.5)

solver.Solver(max_bte_iter=2,multiscale=False,max_fourier_iter=20,load_state=False,save_state=True,max_fourier_error=1e-3)

#plot.Plot(variable='vtk',repeat_x=3,repeat_y=3)
#plot.Plot(variable='map/flux/magnitude',repeat_x=3,repeat_y=3)
