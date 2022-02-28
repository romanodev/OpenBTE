
def first_guess(geometry,material,options_first_guess)->'first_guess':

   import openbte.fourier as fourier
   import numpy as np
   import pickle
   from mpi4py import MPI
   import openbte.utils as utils
   comm = MPI.COMM_WORLD

   data = None  
   if comm.rank == 0:

    argv = {'geometry':geometry,'material':material,'additional_heat_source':options_first_guess.setdefault('additional_heat_source',np.zeros_like(geometry['generation'])),'verbose':options_first_guess.setdefault('verbose',True)} #this needs to change

    f = fourier.solve_fourier_single(**argv)

    data =  {'Temperature_Fourier':f['temperature'],'kappa_fourier':np.array([f['meta'][0]]),'Flux_Fourier':f['flux'],'gradT':f['grad'],'Heat_Generation':geometry['generation']}

   return utils.create_shared_memory_dict(data)



