import openbte.material as material
import openbte.solver as solver
import openbte.geometry as geometry
import openbte.plot as plot


material.Material(model='gray',matfile='TDP.dat',mfp=[1],n_phi=48,n_theta=48,n_mfp=50)

#geometry.Geometry(model='porous/custom',polygons=[[0.25,0.25,0.75,0.25,0.5,1.1]],lx=10,ly=10,lz=0,step=1)
geometry.Geometry(model='porous/custom',polyfile='polygons.dat',lx=10,ly=10,lz=0,step=1)

solver.Solver(max_bte_iter=4,multiscale=False,max_fourier_iter=20,load_state=False,max_fourier_error=1e-3)

plot.Plot(variable='map/temperature',direction='x',repeat_x = 1,repeat_y =1)

