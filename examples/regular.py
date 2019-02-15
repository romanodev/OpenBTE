import openbte.material as material
import openbte.solver as solver
import openbte.geometry as geometry
import openbte.plot as plot


material.Material(model='gray',matfile='TDP.dat',mfp=[1],n_phi=48,n_theta=48,n_mfp=50)

#geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=3,step=3)
geometry.Geometry(type='bulk',porosity=0.25,shape='square',lx=10,ly=10,lz=0,step=1.0)

solver.Solver(max_bte_iter=4,multiscale=False,max_fourier_iter=20,load_state=False,max_fourier_error=1e-3)

plot.Plot(variable='vtk',repeat_x = 1,repeat_y =1)

