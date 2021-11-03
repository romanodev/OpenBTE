
def first_guess(geometry,material,options_first_guess)->'temperatures':

   import openbte.fourier as fourier
   import numpy as np
   import pickle
   from mpi4py import MPI
   import openbte.utils as utils
   comm = MPI.COMM_WORLD

   data = None  
   if comm.rank == 0:

    argv = {'geometry':geometry,'material':material} #this needs to change

    f = fourier.solve_fourier_single(argv)

    data =  {'data':f['temperature'],'kappa':[f['meta'][0]],'flux':f['flux'],'gradT':f['grad']}
    
   return utils.create_shared_memory_dict(data)



