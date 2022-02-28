import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import os
from .utils import *
from .viewer import *
from .kappa_mode import *
from .suppression import *
from .line_data import*
from .vtu import *
from scipy.spatial import distance
import numpy.ma as ma
from mpi4py import MPI
comm = MPI.COMM_WORLD



def Plot(**argv):

   #import data-------------------------------
   geometry     = argv['geometry'] if 'geometry' in argv.keys() else utils.load('geometry')
   solver       = argv['solver']   if 'geometry' in argv.keys() else utils.load('solver')
   material     = argv['material'] if 'material' in argv.keys() else utils.load('material')

   dim = int(geometry['meta'][2])
   model = argv['model']
   
   if model == 'maps':

     output = plot_results(solver,geometry,material,argv)
     if comm.rank == 0 and argv.setdefault('save',True):
      save('sample',output)

     return output

   elif model == 'line':

     line_data = compute_line_data(solver,geometry,argv)

     if argv.setdefault('show',False):
        plot_line_data(line_data) 

     if comm.rank == 0:
      save_data('line_data',line_data)   
      return line_data     

   elif model == 'vtu':

     vtu(material,solver,geometry,argv)

   elif model == 'suppression':

      s = suppression(solver,material,argv) 
      if comm.rank == 0:
        save_data(argv.setdefault('suppression_file','suppression'),{'mfp':material['mfp_sampled'],'suppression':s['Sm']})

   elif model == 'kappa_mode':

      dim = int(geometry['meta'][2])
      #rta = load_data(argv.setdefault('rta_material_file','rta'))

      if dim == 2:
        output = kappa_mode_2DSym(material,solver)
      else:
        output = kappa_mode_3D(material,solver)

      if argv.setdefault('save',True) and comm.rank == 0:
        save_data(argv.setdefault('kappa_mode_file','kappa_mode'),output)
      
      if argv.setdefault('show',False) :
         plot_kappa_mode(output)

      return output  







