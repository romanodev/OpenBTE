import numpy as np
import os
import math
from .database import *
from .full_model import generate_full
from .utils import *
from .mfp2DSym import *
from .mfp3D import *
from .rta2DSym import *
from .rta3D import *
import deepdish as dd
from mpi4py import MPI
import shutil


comm = MPI.COMM_WORLD

class Material(object):

 def __init__(self,**argv):

  if comm.rank == 0:  
   
   save = True   
   model = argv['model']
   if comm.rank == 0:
    if   model == 'unlisted':
      download_file(argv['file_id'],'material.h5')
      save = False

    elif model == 'database':
      #download_file(db['entry_name'],'material.h5')
      source = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/openbte/materials/' + argv['filename'] + '_' + str(argv['temperature']) +'.h5'
      shutil.copyfile(source,os.getcwd() + '/material.h5')
      save = False

    elif model == 'full':
      data = generate_full(**argv)

    elif model == 'mfp2DSym':
      data = generate_mfp2DSym(**argv)

    elif model == 'mfp3D':
      data = generate_mfp3D(**argv)

    elif model == 'rta2DSym':
      data = generate_rta2DSym(**argv)

    elif model == 'rta3D':
      data = generate_rta3D(**argv)

    elif model == 'mfp_ms':
      data = generate_mfp_ms(**argv)

   if save:
      #np.save('material',data)
      dd.io.save('material.h5',data)

