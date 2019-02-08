import openbte.material as material
import openbte.solver as solver
import openbte.geometry as geometry
import openbte.plot as plot



#material.Material(matfile='TDP.dat',n_mfp=100,n_phi=48,n_theta=48)

#geometry.Geometry(type='porous/square_lattice',porosity=0.25,shape='square',lx=10,ly=10,lz=0,step=0.5)

#solver.Solver(max_bte_iter=1,multiscale=False,max_fourier_iter=20,load_state=False,max_fourier_error=1e-3)

plot.Plot(variable='map/flux',direction='x',repeat_x = 2,repeat_y =2)

