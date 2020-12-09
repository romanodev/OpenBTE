from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
from mpi4py import MPI
import scipy.sparse as sp
import time
from scipy.sparse.linalg import lgmres
import sys
from shapely.geometry import LineString
import profile
import scipy
from .solve_mfp import *
from .solve_rta import *
from .solve_full import *
import pkg_resources  
from .fourier import *

comm = MPI.COMM_WORLD

class Solver(object):

  def __init__(self,**argv):

        #COMMON OPTIONS------------
        self.data = argv
        self.tt = np.float64
        self.state = {}
        self.multiscale = argv.setdefault('multiscale',False)
        #self.umfpack = argv.setdefault('umfpack',False)
        self.error_multiscale_fourier = argv.setdefault('multiscale_error_fourier',5e-2)
        self.error_multiscale_ballistic = argv.setdefault('multiscale_error_ballistic',1e-2)
        self.bundle = argv.setdefault('bundle',False)
        self.verbose = argv.setdefault('verbose',True)

        self.save_state = argv.setdefault('save_state',False)
        self.load_state = argv.setdefault('load_state',False)
        self.boost = argv.setdefault('boost',1.0)
        self.relaxation_factor = argv.setdefault('alpha',1.0)
        self.keep_lu = argv.setdefault('keep_lu',True)
        self.only_fourier = argv.setdefault('only_fourier',False)
        self.max_bte_iter = argv.setdefault('max_bte_iter',20)
        #self.min_bte_iter = argv.setdefault('min_bte_iter',20)
        self.max_bte_error = argv.setdefault('max_bte_error',1e-3)
        self.max_fourier_iter = argv.setdefault('max_fourier_iter',20)
        self.deviational = argv.setdefault('deviational',False)
        self.max_fourier_error = argv.setdefault('max_fourier_error',1e-5)
        self.init = False
        self.MF = {}
        #----------------------------
         
        if comm.rank == 0:
         if self.verbose: 
            self.print_logo()
            print('                         SYSTEM                 ',flush=True)   
            print(colored(' -----------------------------------------------------------','green'),flush=True)
            self.print_options()

        #-----IMPORT MESH--------------------------------------------------------------
        if comm.rank == 0:
         
         data = argv['geometry'] if 'geometry' in argv.keys() else load_data('geometry')


         self.n_elems = int(data['meta'][0])
         im = np.concatenate((data['i'],list(np.arange(self.n_elems))))
         jm = np.concatenate((data['j'],list(np.arange(self.n_elems))))
         data['im'] = im
         data['jm'] = jm
        else: data = None
        #self.__dict__.update(create_shared_memory_dict(data))
        self.mesh = create_shared_memory_dict(data)
        self.n_elems = int(self.mesh['meta'][0])
        self.kappa_factor = self.mesh['meta'][1]
        self.dim = int(self.mesh['meta'][2])
        self.n_nodes = int(self.mesh['meta'][3])
        self.n_active_sides = int(self.mesh['meta'][4])
        if comm.rank == 0 and self.verbose: self.mesh_info() 
        if comm.rank == 0 and self.verbose: self.mpi_info() 
        argv['mesh'] = self.mesh
        #------------------------------------------------------------------------------

        #IMPORT MATERIAL---------------------------
        if os.path.isfile('material.npz'):
          names = {0:'material'}  
        elif os.path.isfile('mat_0.npz'):
           names = {0:'mat_0',1:'mat_1'}  
        else:   
           names = {0:'dummy'} #this is when we read it from within the script
      
        self.mat = {}
        for key,value in names.items():    
         if comm.rank == 0:   
          #data = argv['material'].data if 'material' in argv.keys() else dd.io.load(value)

          if 'material' in argv.keys():
            data = argv['material']
          else:            
            data = load_data(value)  
       
          data['kappa'] = data['kappa'][0:self.dim,0:self.dim] 
          if data['model'][0] == 10 : #full
           data['B'] = np.einsum('i,ij->ij',data['scale'],data['B'].T + data['B'])
         else: data = None

         self.mat.update({key:create_shared_memory_dict(data)})
         self.set_model(self.mat[key]['model'][0])
        argv['mat'] = self.mat[0]
        #----------------------

        #Build elem_kappa_mapi
        self.elem_kappa_map = np.array([ list(self.mat[i]['kappa'])  for i in self.mesh['elem_mat_map']])
        self.mesh['elem_kappa_map'] = self.elem_kappa_map


        #self.elem_kappa_map = np.array([ list(np.eye(2)*self.mat[i]['kappa'][0,0])  for i in self.mesh['elem_mat_map']])

        if comm.rank == 0 and self.verbose: self.bulk_info()

        if self.model[0:3] == 'mfp' or self.model[0:3] == 'rta' or self.model[0:3] == 'Gra':
         self.n_serial = self.mat[0]['tc'].shape[0]  
         self.n_parallel = self.mat[0]['tc'].shape[1]  
         block =  self.n_serial//comm.size
         self.ff = range(block*comm.rank,self.n_serial) if comm.rank == comm.size-1 else range(block*comm.rank,block*(comm.rank+1))
         if comm.rank == 0 and self.verbose: self.mfp_info()

        elif self.model == 'Fourier':
          argv['only_fourier'] = True 
          self.only_fourier = True

        elif self.model == 'full': 
         self.n_parallel = len(self.mat[0]['tc'])
         if comm.rank == 0 and self.verbose: self.full_info() 
        #------------------------------------------

        if comm.rank == 0:
            if self.verbose: 
                print(colored(' -----------------------------------------------------------','green'),flush=True)
                print(" ",flush=True)

        if comm.rank == 0:
          argv['cache'] = {}  
          data = solve_fourier_single(self.elem_kappa_map,**argv)


          variables = {0:{'name':'Temperature Fourier','units':'K','data':data['temperature_fourier'],'increment':-self.mesh['applied_gradient']},\
                       1:{'name':'Flux Fourier'       ,'units':'W/m/m','data':data['flux_fourier'],'increment':[0,0,0]}}

          self.state.update({'variables':variables,\
                           'kappa_fourier':data['meta'][0]})
          if self.verbose: self.fourier_info(data)
        else: data = None
        self.fourier = create_shared_memory_dict(data)

        #----------------------------------------------------------------
        if not self.only_fourier:
         #-------SET Parallel info----------------------------------------------
         block =  self.n_parallel//comm.size
         self.rr = range(block*comm.rank,self.n_parallel) if comm.rank == comm.size-1 else range(block*comm.rank,block*(comm.rank+1))

         #-------SOLVE BTE------i
         argv['n_elems'] = len(self.mesh['elems'])
         argv['fourier'] = self.fourier
         argv['dim'] = self.dim
         argv['rr'] = self.rr
         argv['n_parallel'] = self.n_parallel
         argv['kappa_factor'] = self.kappa_factor
         
         if argv.setdefault('user',False):

            argv['n_serial'] = self.n_serial

            data = argv['user_model'].solve(**argv)

         elif self.model[0:3] == 'mfp' or \
            self.model[0:3] == 'Gra' :
            argv['n_serial'] = self.n_serial
            data = solve_mfp(**argv) 

         elif self.model[0:3] == 'rta':
            argv['n_serial'] = self.n_serial
            data = solve_rta(**argv) 

         elif self.model == 'full':
            data = solve_full(**argv) 


         #self.state.update({'kappa':data['kappa_vec']})
         #-----------------------

        #Saving-----------------------------------------------------------------------
        if comm.rank == 0:
          variables = self.state['variables']

          if not self.only_fourier:
           variables[2]    = {'name':'Temperature BTE','units':'K','data':data['temperature'],'increment':-self.mesh['applied_gradient']}
           variables[3]    = {'name':'Flux BTE'       ,'units':'W/m/m','data':data['flux'],'increment':[0,0,0]}
           if 'pseudo' in data.keys():
            variables[4]    = {'name':'Gradient Pseudo','units':'K/m','data':data['pseudo'],'increment':[0,0,0]}
           if 'GradT' in data.keys():
            variables[5]    = {'name':'GradT','units':'K/m','data':data['GradT'],'increment':[0,0,0]}
           if 'DeltaT' in data.keys():
            variables[6]    = {'name':'DeltaT','units':'K/m','data':data['DeltaT'],'increment':[0,0,0]}

           #self.state.update({'kappa':data['kappa_vec']})
           self.state.update(data)

          if argv.setdefault('save',True):
           #if self.bundle:
           #  self.state['geometry'] = self.mesh
           #  self.state['material'] = self.mat
           #dd.io.save('solver.h5',self.state)

           save_data(argv.setdefault('filename','solver'),self.state)   

          if self.verbose:
           print(' ',flush=True)   
           print(colored('                 OpenBTE ended successfully','green'),flush=True)
           print(' ',flush=True)  


  def mpi_info(self):

          print(colored('  vCPUs:                                   ','green') + str(comm.size),flush=True)


  def fourier_info(self,data):

          print('                        FOURIER                 ',flush=True)   
          print(colored(' -----------------------------------------------------------','green'),flush=True)
          print(colored('  Iterations:                              ','green') + str(int(data['meta'][2])),flush=True)
          print(colored('  Relative error:                          ','green') + '%.1E' % (data['meta'][1]),flush=True)
          print(colored('  Fourier Thermal Conductivity [W/m/K]:    ','green') + str(round(data['meta'][0],3)),flush=True)
          print(colored(' -----------------------------------------------------------','green'),flush=True)
          print(" ")


  def print_options(self):
          print(colored('  Multiscale:                              ','green')+ str(self.multiscale),flush=True)
          print(colored('  Relaxation:                              ','green')+ str(self.relaxation_factor),flush=True)
          print(colored('  Keep Lu:                                 ','green')+ str(self.keep_lu),flush=True)
          #print(colored('  Use umfpack                              ','green')+ str(self.umfpack),flush=True)
          #print(colored('  Deviational                              ','green')+ str(self.deviational),flush=True)
          print(colored('  Load State                               ','green')+ str(self.load_state),flush=True)
          print(colored('  Save State                               ','green')+ str(self.save_state),flush=True)
          print(colored('  Multiscale Error (Fourier):              ','green')+ str(self.error_multiscale_fourier),flush=True)
          print(colored('  Multiscale Error (Ballistic):            ','green')+ str(self.error_multiscale_ballistic),flush=True)
          print(colored('  Only Fourier:                            ','green')+ str(self.only_fourier),flush=True)
          print(colored('  Max Fourier Error:                       ','green')+ '%.1E' % (self.max_fourier_error),flush=True)
          print(colored('  Max Fourier Iter:                        ','green')+ str(self.max_fourier_iter),flush=True)
          print(colored('  Max BTE Error:                           ','green')+ '%.1E' % (self.max_bte_error),flush=True)
          #print(colored('  Min BTE Iter:                            ','green')+ str(self.min_bte_iter),flush=True)
          print(colored('  Max BTE Iter:                            ','green')+ str(self.max_bte_iter),flush=True)

  def full_info(self):
          print(colored('  Number of modes:                         ','green')+ str(self.n_parallel),flush=True)

  def mfp_info(self):
          print(colored('  Number of MFP:                           ','green')+ str(self.n_serial),flush=True)
          print(colored('  Number of Solid Angles:                  ','green')+ str(self.n_parallel),flush=True)

  def bulk_info(self):

          #print('                        MATERIAL                 ')   
          #print(colored(' -----------------------------------------------------------','green'))
          print(colored('  Model:                                   ','green')+ self.model,flush=True)

          for key,value in self.mat.items():
           print(colored('  Bulk Thermal Conductivity (xx) [W/m/K]:  ','green')+ str(round(self.mat[key]['kappa'][0,0],2)),flush=True)
           print(colored('  Bulk Thermal Conductivity (yy) [W/m/K]:  ','green')+ str(round(self.mat[key]['kappa'][1,1],2)),flush=True)
           if self.dim == 3:
            print(colored('  Bulk Thermal Conductivity (zz)[W/m/K]:   ','green')+ str(round(self.mat[key]['kappa'][2,2],2)),flush=True)

           dirr = self.mesh['meta'][-1]
           if dirr == 0:
               direction = 'x'
           elif dirr == 1:    
               direction = 'y'
           else:    
               direction = 'z'
           print(colored('  Applied Thermal Gradient along :         ','green')+ direction,flush=True)

          #print(colored(' -----------------------------------------------------------','green'))
          #print(" ")


  def mesh_info(self):

          #print('                        SPACE GRID                 ')   
          #print(colored(' -----------------------------------------------------------','green'))
          print(colored('  Dimension:                               ','green') + str(self.dim),flush=True)
          print(colored('  Size along X [nm]:                       ','green')+ str(round(self.mesh['size'][0],2)),flush=True)
          print(colored('  Size along y [nm]:                       ','green')+ str(round(self.mesh['size'][1],2)),flush=True)
          if self.dim == 3:
           print(colored('  Size along z [nm]:                       ','green')+ str(round(self.mesh['size'][2],2)),flush=True)
          print(colored('  Number of Elements:                      ','green') + str(self.n_elems),flush=True)
          print(colored('  Number of Sides:                         ','green') + str(len(self.mesh['active_sides'])),flush=True)
          print(colored('  Number of Nodes:                         ','green') + str(len(self.mesh['nodes'])),flush=True)

          if self.dim == 3:
           filling = np.sum(self.mesh['volumes'])/self.mesh['size'][0]/self.mesh['size'][1]/self.mesh['size'][2]
          else: 
           filling = np.sum(self.mesh['volumes'])/self.mesh['size'][0]/self.mesh['size'][1]
          print(colored('  Computed porosity:                       ','green') + str(round(1-filling,3)),flush=True)
          
          #print(colored(' -----------------------------------------------------------','green'))
          #print(" ")



  def set_model(self,m):

    if   m == 0:
     self.model = 'Fourier'

    elif m == 1:
     self.model = 'Gray2D'

    elif m == 2:
     self.model = 'gray2Dsym'

    elif m == 3:
     self.model = 'gray3D'

    elif m == 4:
     self.model = 'mfp2D'

    elif m == 5:
     self.model = 'mfp2DSym'

    elif m == 6:
     self.model = 'mfp3D'

    elif m == 7:
     self.model = 'rta2D'

    elif m == 8:
     self.model = 'rta2DSym'

    elif m == 9:
     self.model = 'rta3D'

    elif m == 10:
     self.model = 'full'
    else: 
     print('No model recognized')
     quit()



  def print_logo(self):


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
    print('                       WARNINGS',flush=True)
    print(colored(' -----------------------------------------------------------','green'),flush=True)
    print(flush=True)   
