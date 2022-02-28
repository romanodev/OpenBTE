import numpy as np
import h5py
import sys
import openbte.utils as utils
from mpi4py import MPI

comm = MPI.COMM_WORLD

def almabte2openbte(options_almabte)->'rta':
 data = None

 if comm.rank == 0:
  filename = options_almabte['phonon_info']
 
  tmp = np.loadtxt(filename,skiprows=1,delimiter=',')
  nq = int(np.max(tmp[:,0]))+1
  nb = int(np.max(tmp[:,1]))+1
  ntot = nq*nb

  tau = tmp[:,7 ]
  v   = tmp[:,8:]
  C   = tmp[:,6 ]
  w   = tmp[:,5 ]
  kappa =  np.einsum('ki,kj,k,k->ij',v,v,tau,C)

  data = {'heat_capacity':C,\
         'scattering_time':tau,\
         'group_velocity':v,\
         'frequency':w/2.0/np.pi,\
         'thermal_conductivity':kappa}

 return utils.create_shared_memory_dict(data)




def main():

   almabte_options = {'phonon_info':sys.argv[1]}

   data = almabte2openbte(almabte_options)

   if len(sys.argv) == 3:
     output = sys.argv[2]
   else  :
     output = 'rta'

   utils.save_data(output,data)   

