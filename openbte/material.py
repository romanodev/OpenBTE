import numpy as np
import os
import math
from .database import *
from .full_model import generate_full
from .utils import *
from .mfp2DSym_2 import *
from .mfp2DSym import *
from .mfp3D import *
from .mfp3D_2 import *
from .gray2D import *
from .rta2DSym import *
from .rta3D import *
#import deepdish as dd
from mpi4py import MPI
import shutil


comm = MPI.COMM_WORLD

class Material(object):

 def __init__(self,**argv):

  #if comm.rank == 0:  
   
   save = True   
   model = argv['model']
   if model == 'unlisted':
    if comm.rank == 0:
      download_file(argv['file_id'],'material.h5')
      save = False

   elif model == 'database':
    if comm.rank == 0:
      source = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/openbte/materials/' + argv['filename'] + '_' + str(argv['temperature']) +'.npz'
      shutil.copyfile(source,os.getcwd() + '/material.npz')
      save = False

   elif model == 'full':
     if comm.rank == 0:
      data = generate_full(**argv)

   elif model == 'mfp2DSym':
      #data = generate_mfp2DSym(**argv)
      data = generate_mfp2DSym_2(**argv)
    
   elif model == 'gray2D':
     if comm.rank == 0:
      data = generate_gray2D(**argv)

   elif model == 'mfp3D':
      # if comm.rank == 0:
      #data = generate_mfp3D(**argv)
      data = generate_mfp3D_2(**argv)

   elif model == 'rta2DSym':
     if comm.rank == 0:
      data = generate_rta2DSym(**argv)

   elif model == 'rta3D':
     if comm.rank == 0:
      data = generate_rta3D(**argv)

   elif model == 'mfp_ms':
     if comm.rank == 0:
      data = generate_mfp_ms(**argv)

   if save:
     if comm.rank == 0:
      np.savez_compressed(argv.setdefault('filename','material') + '.npz',data)   

