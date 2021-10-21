from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
import openbte.utils as utils
from mpi4py import MPI
import sys
import scipy.sparse as sp
import time
from scipy.sparse.linalg import lgmres
import sys
from shapely.geometry import LineString
import scipy
from .solve_rta import *
import pkg_resources  
from .first_guess import first_guess
import warnings

comm = MPI.COMM_WORLD

def print_bulk_info(mat,mesh):

           #print(colored('  Model:                                   ','green')+ get_model(mat['model']),flush=True)
          
           if mat['kappa'].ndim == 3:
              for kappa in mat['kappa']:    
                print(colored('  Bulk Thermal Conductivity (xx) [W/m/K]:  ','green')+ str(round(kappa[0,0],2)),flush=True)
                print(colored('  Bulk Thermal Conductivity (yy) [W/m/K]:  ','green')+ str(round(kappa[1,1],2)),flush=True)
                if self.dim == 3:
                  print(colored('  Bulk Thermal Conductivity (zz)[W/m/K]:   ','green')+ str(round(kappa[2,2],2)),flush=True)
           else: 
                kappa = mat['kappa']
                print(colored('  Bulk Thermal Conductivity (xx) [W/m/K]:  ','green')+ str(round(kappa[0,0],2)),flush=True)
                print(colored('  Bulk Thermal Conductivity (yy) [W/m/K]:  ','green')+ str(round(kappa[1,1],2)),flush=True)
                if len(kappa) == 3:
                  print(colored('  Bulk Thermal Conductivity (zz)[W/m/K]:   ','green')+ str(round(kappa[2,2],2)),flush=True)

           dirr = ['x','y','z']
           print(colored('  Applied Thermal Gradient along :         ','green')+ dirr[int(mesh['meta'][-2])],flush=True)


def print_options(**argv):
          print(colored('  Keep Lu:                                 ','green')+ str(argv['keep_lu']),flush=True)
          print(colored('  Only Fourier:                            ','green')+ str(argv['only_fourier']),flush=True)
          print(colored('  Max Fourier Error:                       ','green')+ '%.1E' % (argv['max_fourier_error']),flush=True)
          print(colored('  Max Fourier Iter:                        ','green')+ str(argv['max_fourier_iter']),flush=True)
          print(colored('  Max BTE Error:                           ','green')+ '%.1E' % (argv['max_bte_error']),flush=True)
          print(colored('  Max BTE Iter:                            ','green')+ str(argv['max_bte_iter']),flush=True)


def print_logo():

    v = pkg_resources.require("OpenBTE")[0].version   
    print(' ',flush=True)
    print(colored(r'''        ___                   ____ _____ _____ ''','green'),flush=True)
    print(colored(r'''       / _ \ _ __   ___ _ __ | __ )_   _| ____|''','green'),flush=True)
    print(colored(r'''      | | | | '_ \ / _ \ '_ \|  _ \ | | |  _|  ''','green'),flush=True)
    print(colored(r'''      | |_| | |_) |  __/ | | | |_) || | | |___ ''','green'),flush=True)
    print(colored(r'''       \___/| .__/ \___|_| |_|____/ |_| |_____|''','green'),flush=True)
    print(colored(r'''            |_|                                ''','green'),flush=True)
    print(colored(r'''                                               v. ''' + str(v) ,'blue'),flush=True)
    print(flush=True)
    print('                       GENERAL INFO',flush=True)
    print(colored(' -----------------------------------------------------------','green'),flush=True)
    print(colored('  Contact:          ','green') + 'romanog@mit.edu                        ',flush=True) 
    print(colored('  Source code:      ','green') + 'https://github.com/romanodev/OpenBTE     ',flush=True)
    print(colored('  Become a sponsor: ','green') + 'https://github.com/sponsors/romanodev    ',flush=True)
    print(colored('  Documentation:    ','green') + 'https://openbte.readthedocs.io',flush=True)
    print(colored(' -----------------------------------------------------------','green'),flush=True)
    print(flush=True)   



def Solver(**argv):

        #COMMON OPTIONS--------------------------------------
        verbose      = argv.setdefault('verbose',True)
        only_fourier = argv.setdefault('only_fourier',False)
        #---------------------------------------------------

        geometry = argv['geometry'] if 'geometry' in argv.keys() else utils.load_shared('geometry')
        material = argv['material'] if 'material' in argv.keys() else utils.load_shared('material')

        #Print relevant options-----------------------------
        if comm.rank == 0 :
            if verbose:
             print_logo()
             print('                         SYSTEM                 ',flush=True)   
             print(colored(' -----------------------------------------------------------','green'),flush=True)
             print_bulk_info(material,geometry)
             print_mpi_info()
             print_grid_info(geometry,material)
        #---------------------------------------------------

        #Solve fourier--
        X = first_guess(geometry,material,{})
        
        #Solve bte--
        if not only_fourier:
           output = solve_rta(geometry,material,X,argv)

        #prepare_data(argv)
        if comm.rank == 0 and argv.setdefault('save',True):
           utils.save_data(argv.setdefault('filename','solver'),output)   

        if argv['verbose'] and comm.rank == 0:
         print(' ',flush=True)   
         print(colored('                 OpenBTE ended successfully','green'),flush=True)
         print(' ',flush=True)  

        return output


def print_mpi_info():

          print(colored('  vCPUs:                                   ','green') + str(comm.size),flush=True)
          print(" ")



def print_grid_info(mesh,mat):

          dim     = int(mesh['meta'][2])
          n_elems = len(mesh['elems'])
          print('                          GRID                 ')   
          print(colored(' -----------------------------------------------------------','green'))
          print(colored('  Dimension:                               ','green') + str(dim),flush=True)
          print(colored('  Size along X [nm]:                       ','green')+ str(round(mesh['size'][0],2)),flush=True)
          print(colored('  Size along y [nm]:                       ','green')+ str(round(mesh['size'][1],2)),flush=True)
          if dim == 3:
           print(colored('  Size along z [nm]:                       ','green')+ str(round(mesh['size'][2],2)),flush=True)
          print(colored('  Number of Elements:                      ','green') + str(n_elems),flush=True)
          print(colored('  Number of Sides:                         ','green') + str(len(mesh['active_sides'])),flush=True)
          print(colored('  Number of Nodes:                         ','green') + str(len(mesh['nodes'])),flush=True)


          with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            print(colored('  Number of MFP:                           ','green')+ str(mat['tc'].shape[0]),flush=True)
            print(colored('  Number of Solid Angles:                  ','green')+ str(mat['tc'].shape[1]),flush=True)

          if dim == 3:
           filling = np.sum(mesh['volumes'])/mesh['size'][0]/mesh['size'][1]/mesh['size'][2]
          else: 
           filling = np.sum(mesh['volumes'])/mesh['size'][0]/mesh['size'][1]
          print(colored('  Computed porosity:                       ','green') + str(round(1-filling,3)),flush=True)
          
          print(colored(' -----------------------------------------------------------','green'))
          print(" ")






