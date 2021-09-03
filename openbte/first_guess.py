
import openbte.fourier as fourier
import numpy as np
import pickle
from mpi4py import MPI
import openbte.utils as utils

comm = MPI.COMM_WORLD

def first_guess(geometry,material,options_first_guess)->'temperatures':

   data = None  
   if comm.rank == 0:

    argv = {'geometry':geometry,'material':material} #this needs to change

    f = fourier.solve_fourier_single(argv)

    if options_first_guess.setdefault('add_gradient',False):
     X = f['temperature'][np.newaxis,:]-np.einsum('uj,cj->uc',material['F'],f['grad'])*1e9*0
    else:
     X = f['temperature']

    data =  {'data':X,'kappa':[f['meta'][0]],'flux':f['flux']}
    
   return utils.create_shared_memory_dict(data)



