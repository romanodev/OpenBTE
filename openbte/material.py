import numpy as np
import os
import math
#from .database import *
from .full_model import generate_full
from .utils import *
from .mfp2DSym import *
from .mfp2D import *
from .mfp3D import *
from .gray2D import *
from .rta2DSym import *
from .rta3D import *
from mpi4py import MPI
import shutil
import pickle
import lzma
import gzip
import bz2
#import BytesIO


comm = MPI.COMM_WORLD

class Material(object):

 def __init__(self,**argv):

   save = argv.setdefault('save',True)
   model = argv['model']
   source = argv.setdefault('source','local')
   if source == 'database':
    if comm.rank == 0:
        basename = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + \
                  '/openbte/materials/' + model[:3] + '_' +  argv['filename'] + \
                  '_' + str(argv['temperature']) 

        argv['basename'] = basename

        #shutil.copyfile(filename,os.getcwd() + '/' + model[:3] +'.npz')
   if source == 'unlisted':
    if comm.rank == 0:
      download_file(argv['file_id'],'rta.npz')

   if model == 'full':
     if comm.rank == 0:
      data = generate_full(**argv)

   elif model == 'mfp2DSym':
      data = generate_mfp2DSym(**argv)
   
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
      data = generate_mfp2D(**argv)

   elif model == 'gray2DSym':
      data = generate_mfp2DSym(**argv)

   elif model == 'gray2D':
      data = generate_gray2D(**argv)

   elif model == 'gray3D':
      data = generate_mfp3D(**argv)

   elif model == 'mfp3D':
      data = generate_mfp3D(**argv)

   elif model == 'rta2DSym':
     if comm.rank == 0:
      data = generate_rta2DSym(**argv)

   elif model == 'rta3D':
     if comm.rank == 0:
      data = generate_rta3D(**argv)

   if save :
     if comm.rank == 0:
         save_data(argv.setdefault('output_filename','material'),data)   
   else:
     if comm.rank == 0:
      self.state = data
