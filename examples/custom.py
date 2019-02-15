import openbte.material as material
import openbte.solver as solver
import openbte.geometry as geometry
import openbte.plot as plot


#material.Material(model='gray',matfile='TDP.dat',mfp=[1],n_phi=48,n_theta=48,n_mfp=50)

geometry.Geometry(type='porous/custom',polyfile='polygons.dat',lx=10,ly=10,lz=2,step=2)

#solver.Solver(max_bte_iter=4,multiscale=False,max_fourier_iter=20,load_state=False,max_fourier_error=1e-3)

#plot.Plot(variable='map/temperature',direction='x',repeat_x = 1,repeat_y =1)

