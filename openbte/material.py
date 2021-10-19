import numpy as np
import os
import math
from .utils import *
#from .mfp2DSym import *
#from .mg2DSym import *
#from .mfp2D import *
#from .mfp3D import *
from .gray2D import *
from .gray2DSym import *
from .rta2DSym import *
from .rta3D import *
from mpi4py import MPI
import shutil

comm = MPI.COMM_WORLD

def database(database_material)->'rta':
   data = None 

   if comm.rank == 0:
     filename = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + \
                  '/openbte/materials/' +  database_material
   
     data = load_data(filename)

   return  create_shared_memory_dict(data)

def Material(**argv):

  argv.setdefault('filename','rta')  
  if 'custom' in argv.keys():
    data = argv['custom'](argv)
  else:  

   model = argv.setdefault('model','rta3D')

   #set up database
   if not model == 'gray':
    source = argv.setdefault('source','database')
    if source == 'database':
         rta = database(argv['filename'])  
    else :     
         rta = load_data(argv['filename'])  

   if model == 'rta2DSym':
      data  = rta2DSym(rta,argv)

   elif model == 'rta3D':
      data  = rta3D(rta,argv)

   elif model == 'fourier':
      
      kappa = np.eye(3)
      if 'kappa' in argv.keys():
        kappa *= argv['kappa']
      else:  
       kappa[0,0] = argv['kappa_xx']
       kappa[1,1] = argv['kappa_yy']
       kappa[2,2] = argv['kappa_zz']
      data = {'kappa':kappa,'model':[0]}

   elif model == 'gray':
      data = gray2D(argv)

   else:   
      print('No model recognized')
      quit()


  if argv.setdefault('save',True):
     if comm.rank == 0:
         save_data(argv.setdefault('output_filename','material'),data)   

  return data

 



