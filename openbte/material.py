import numpy as np
import os
import math
from .database import *
from .full_model2 import generate_full
from .utils import *
from .mfp2DSym import *
from .mfp import *
import deepdish as dd
from mpi4py import MPI

comm = MPI.COMM_WORLD

class Material(object):


 def __init__(self,**argv):

  if comm.rank == 0:  
     
   model = argv['model']
   if comm.rank == 0:
    if   model == 'unlisted':
      download_file(argv['file_id'],'material.h5')

    elif model == 'database':
      download_file(db['entry_name'],'material.h5')

    elif model == 'full':
      dd.io.save('material.h5',generate_full(**argv)) 

    elif model == 'mfp2DSym':
      dd.io.save('material.h5',generate_mfp2DSym(**argv)) 

    elif model == 'mfp':
      dd.io.save('material.h5',generate_mfp(**argv)) 

    elif model == 'mfp_ms':
      dd.io.save('material.h5',generate_mfp_ms(**argv)) 
