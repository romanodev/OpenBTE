from material import *
from geometry import *
from solver import *
from plot import *
import mpi4py		
import time



mat = Material(filename='Si-300K',grid=[50,12,48],delta=0.0,min_mfp=1e-12)


for n in range(5000):


 dirs = os.listdir('confs2')
 n_max = -1
 for d in dirs:
  n_max = max([n_max,int(d.split('_')[1].split('.')[0])])

 m = n_max + 1

 geo = Geometry(model='porous/random',
               frame = [10.0,10.0],
               porosity = 0.15,
               step =2.0,
               shape = 'square',
               Nx = 2,
               Ny = 2,
               save_configuration = True)
#geo = Geometry(model='load')


#geo = Geometry(model='bulk',
#               frame = [100.0,100.0],
#               porosity = 0.25,
#               step = 10.0,
#               shape = 'square')

 #find next n


 if MPI.COMM_WORLD.Get_rank() == 0:
  a = time.time()
  os.system('mv conf.dat ' + 'confs2/conf_' + str(m) + '.dat')
 sol = Solver(model='bte',material=mat,geometry=geo,max_bte_iter = 10,multiscale=False)
 if MPI.COMM_WORLD.Get_rank() == 0:
  print(time.time() - a)
  np.array(sol.state['kappa_bte']).dump(file('kappa.dat','w+'))
  os.system('mv kappa.dat ' + 'confs2/kappa_' + str(m) + '.dat')

#Plot(model='map/flux_magnitude',material=mat,geometry=geo)
#Plot(plot='suppression_function',material=mat,geometry=geo)

























