import numpy as np
import os
import math
from .full_model import *
from .utils import *
from .mfp2DSym import *
from .mg2DSym import *
from .mfp2D import *
from .mfp3D import *
from .gray2D import *
from .gray2DSym import *
from .rta2DSym import *
from .rta3D import *
from mpi4py import MPI
import shutil

comm = MPI.COMM_WORLD

def Material(**argv):

   model = argv.setdefault('model','rta3D')

   #set up database
   source = argv.setdefault('source','database')
   if source == 'database':
    if comm.rank == 0:
        if 'temperature' in argv.keys():
         filename = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + \
                  '/openbte/materials/rta_' +  argv['filename'] + \
                  '_' + str(argv['temperature']) 
        else:          
         filename = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + \
                  '/openbte/materials/' +  argv['filename']  

        argv['filename'] = filename

   if source == 'unlisted':
    if comm.rank == 0:
      download_file(argv['file_id'],'rta.npz')

   if model == 'rta2DSym':
     if comm.rank == 0:
      data = rta2DSym(**argv)

   elif model == 'full':
     if comm.rank == 0:
      data = full(**argv)

   elif model == 'mfp2DSym':
      data = mfp2DSym(**argv)

   elif model == 'fourier':
      
      kappa = np.eye(3)
      if 'kappa' in argv.keys():
        kappa *= argv['kappa']
      else:  
       kappa[0,0] = argv['kappa_xx']
       kappa[1,1] = argv['kappa_yy']
       kappa[2,2] = argv['kappa_zz']
      data = {'kappa':kappa,'model':[0]}

   elif model == 'mfp2D':
      data = mfp2D(**argv)

   elif model == 'gray2DSym':
      data = gray2DSym(**argv)

   elif model == 'mg2DSym':
     if comm.rank == 0:
      data = mg2DSym(**argv)

   elif model == 'gray2D':
      data = gray2D(**argv)

   #elif model == 'gray3D':
   #   data = mfp3D(**argv)
   
   elif model == 'mfp3D':
      data = mfp3D(**argv)

   elif model == 'rta3D':
     if comm.rank == 0:
      data = rta3D(**argv)

   else:   
      print('No model recognized')
      quit()


   if argv.setdefault('save',True):
     if comm.rank == 0:
         save_data(argv.setdefault('output_filename','material'),data)   

   if comm.rank == 0:
    return data

 



