from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
from mpi4py import MPI
import sys
import scipy.sparse as sp
import time
from scipy.sparse.linalg import lgmres
import sys
from shapely.geometry import LineString
import profile
import scipy
from .solve_rta import *
from .solve_full import *
import pkg_resources  
from .fourier import *
import warnings

comm = MPI.COMM_WORLD

def get_model(m):

    models = ['Fourier','gray2D','gray2DSym','gray3D','mfp2D','mfp2DSym','mfp3D','rta2D','rta2DSym','rta3D','full','mg2DSym']


    return models[m[0]]

def fourier_info(argv):
          print('                        FOURIER                 ',flush=True)   
          print(colored(' -----------------------------------------------------------','green'),flush=True)
          #print(colored('  Iterations:                              ','green') + str(int(data[2])),flush=True)
          #print(colored('  Relative error:                          ','green') + '%.1E' % (data[1]),flush=True)
          #print(colored('  Fourier Thermal Conductivity [W/m/K]:    ','green') + str(round(data[0],3)),flush=True)
          print(colored(' -----------------------------------------------------------','green'),flush=True)
          print(" ")



def print_bulk_info(mat,mesh):

           print(colored('  Model:                                   ','green')+ get_model(mat['model']),flush=True)
          
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
           print(colored('  Applied Thermal Gradient along :         ','green')+ dirr[int(mesh['meta'][-1])],flush=True)


def print_options(**argv):
          print(colored('  Multiscale:                              ','green')+ str(argv['multiscale']),flush=True)
          print(colored('  Relaxation:                              ','green')+ str(argv['alpha']),flush=True)
          print(colored('  Keep Lu:                                 ','green')+ str(argv['keep_lu']),flush=True)
          print(colored('  Load State                               ','green')+ str(argv['load_state']),flush=True)
          print(colored('  Save State                               ','green')+ str(argv['save_state']),flush=True)
          print(colored('  Multiscale Error (Fourier):              ','green')+ str(argv['multiscale_error_fourier']),flush=True)
          print(colored('  Multiscale Error (Ballistic):            ','green')+ str(argv['multiscale_error_ballistic']),flush=True)
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


def prepare_data(argv):
        
  #Original Description--
  tmp = {0:{'name':'Temperature Fourier','units':'K','data':argv['fourier']['temperature'],'increment':-argv['geometry']['applied_gradient']},\
               1:{'name':'Flux Fourier'       ,'units':'W/m/m','data':argv['fourier']['flux'],'increment':[0,0,0]}}

  if not argv.setdefault('only_fourier',False):
      tmp[2]    = {'name':'Temperature BTE','units':'K','data':argv['bte']['temperature'],'increment':-argv['geometry']['applied_gradient']}
      tmp[3]    = {'name':'Flux BTE'       ,'units':'W/m/m','data':argv['bte']['flux'],'increment':[0,0,0]}
              
  #Here we unroll variables for later use--
  variables = {}
  for key,value in tmp.items():
      
     #if value['data'].ndim == 1: #scalar
     variables[value['name']] = {'data':value['data'],'units':value['units'],'increment':value['increment']}
     #elif value['data'].ndim == 2 : #vector 
     #    variables[value['name'] + '(x)'] = {'data':value['data'][:,0],'units':value['units'],'increment':value['increment']}
     #    variables[value['name'] + '(y)'] = {'data':value['data'][:,1],'units':value['units'],'increment':value['increment']}
     #    if argv['dim'] == 3: 
     #        variables[value['name'] + '(z)'] = {'data':value['data'][:,2],'units':value['units'],'increment':value['increment']}
     #    mag = np.array([np.linalg.norm(value) for value in value['data']])
     #    variables[value['name'] + '(mag.)'] = {'data':mag,'units':value['units'],'increment':value['increment']}
  argv['variables'] = variables

  dirr = int(argv['geometry']['meta'][-1])
  argv['data'] = {'variables':variables,'kappa_bulk':argv['material']['kappa'][dirr,dirr] ,'kappa_fourier':argv['fourier']['meta'][0]}

  if 'bte' in argv.keys():
     argv['data']['kappa_bte']  = argv['bte']['kappa'][-1]  
     if 'kappa_mode' in argv['bte']:
       argv['data']['kappa_mode'] = argv['bte']['kappa_mode']
     if 'kappa_mode_f' in argv['bte']:
       argv['data']['kappa_mode_f'] = argv['bte']['kappa_mode_f']
     if 'kappa_0' in argv['bte']:
       argv['data']['kappa_0'] = argv['bte']['kappa_0']
     if 'tau' in argv['bte']:
       argv['data']['tau'] = argv['bte']['tau']




def Solver(**argv):


        #COMMON OPTIONS------------
        argv.setdefault('multiscale',False)
        argv.setdefault('multiscale_error_fourier',5e-2)
        argv.setdefault('multiscale_error_ballistic',1e-2)
        argv.setdefault('verbose',True)
        argv.setdefault('save_state',False)
        argv.setdefault('load_state',False)
        argv.setdefault('alpha',1.0)
        argv.setdefault('keep_lu',False)
        argv.setdefault('only_fourier',False)
        argv.setdefault('max_bte_iter',20)
        argv.setdefault('max_bte_error',1e-3)
        argv.setdefault('max_fourier_iter',20)
        argv.setdefault('deviational',False)
        argv.setdefault('max_fourier_error',1e-5)
        #----------------------------
       

        mesh = None 
        mat = None 
        if comm.rank == 0 :
            if argv['verbose']:
             print_logo()
             print('                         SYSTEM                 ',flush=True)   
             print(colored(' -----------------------------------------------------------','green'),flush=True)
             print_options(**argv)

            mesh = argv['geometry'] if 'geometry' in argv.keys() else load_data(argv.setdefault('geometry_filename','geometry'))
            mat  = argv['material'] if 'material' in argv.keys() else load_data(argv.setdefault('material_filename','material'))
            
            if argv['verbose']:
             print_bulk_info(mat,mesh)
             print_mpi_info()
             print_grid_info(mesh,mat)

        if comm.size > 1:

         #load geometry---
         argv['geometry'] = create_shared_memory_dict(mesh)
 
         #load material---
         argv['material'] = create_shared_memory_dict(mat)

         argv['geometry']['elem_kappa_map'] = get_kappa_map_from_mat(**argv)   

         #Solve fourier--
         argv['fourier'] = create_shared_memory_dict(solve_fourier_single(argv))

        else:

         #load geometry---
         argv['geometry'] = mesh
 
         #load material---
         argv['material'] = mat

         argv['geometry']['elem_kappa_map'] = get_kappa_map_from_mat(**argv)

         #Solve fourier--
         argv['fourier'] = solve_fourier_single(argv)


        #Solve bte--
        if not argv['only_fourier']:

           mat_model = get_model(argv['material']['model'])

           if mat_model in ['rta2DSym','rta2D','rta3D','mfp2DSym','mfp2D','gray2D','gray2DSym','mg2DSym'] :
               solve_rta(argv)

           elif mat_model == 'full':    
               solve_full(argv)

           elif mat_model == 'exp':    
               solve_exp(argv)

           #elif mat_model in ['mg2DSym']:    
           #    solve_mg(argv)

           else:
               print('No model coded')
               quit()

        
        argv['dim'] =  int(argv['geometry']['meta'][2])
        prepare_data(argv)
        if comm.rank == 0 and argv.setdefault('save',True):
           save_data(argv.setdefault('filename','solver'),argv['data'])   

        #Clear cache
        clear_fourier_cache() 
        clear_BTE_cache() 
        if argv['verbose'] and comm.rank == 0:
         print(' ',flush=True)   
         print(colored('                 OpenBTE ended successfully','green'),flush=True)
         print(' ',flush=True)  


        return argv['data']


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
            if mat['model'] in [8,9]:   
             print(colored('  Number of MFP:                           ','green')+ str(mat['tc'].shape[0]),flush=True)
             print(colored('  Number of Solid Angles:                  ','green')+ str(mat['tc'].shape[1]),flush=True)
            else : 
             print(colored('  Number of wave vectors:                  ','green')+ str(mat['sigma'].shape[0]),flush=True)

          if dim == 3:
           filling = np.sum(mesh['volumes'])/mesh['size'][0]/mesh['size'][1]/mesh['size'][2]
          else: 
           filling = np.sum(mesh['volumes'])/mesh['size'][0]/mesh['size'][1]
          print(colored('  Computed porosity:                       ','green') + str(round(1-filling,3)),flush=True)
          
          print(colored(' -----------------------------------------------------------','green'))
          print(" ")






