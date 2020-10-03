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

comm = MPI.COMM_WORLD

class Solver(object):

  def __init__(self,**argv):

        #COMMON OPTIONS------------
        self.data = argv
        self.tt = np.float64
        self.state = {}
        self.multiscale = argv.setdefault('multiscale',False)
        #self.umfpack = argv.setdefault('umfpack',False)
        self.error_multiscale = argv.setdefault('multiscale_error',1e-2)
        self.bundle = argv.setdefault('bundle',False)
        self.verbose = argv.setdefault('verbose',True)

        self.save_state = argv.setdefault('save_state',False)
        self.load_state = argv.setdefault('load_state',False)
        self.boost = argv.setdefault('boost',1.0)
        self.relaxation_factor = argv.setdefault('alpha',1.0)
        self.keep_lu = argv.setdefault('keep_lu',True)
        self.only_fourier = argv.setdefault('only_fourier',False)
        self.max_bte_iter = argv.setdefault('max_bte_iter',10)
        self.min_bte_iter = argv.setdefault('min_bte_iter',20)
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
         
         data = argv['geometry'].data if 'geometry' in argv.keys() else load_data('geometry')


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
            data = argv['material'].data
          else:            
            data = load_data(value)  
        
          
          #data['VMFP']  *= 1e9
          #data['sigma'] *= 1e9
          data['kappa'] = data['kappa'][0:self.dim,0:self.dim] 
          if data['model'][0] == 10 : #full
           data['B'] = np.einsum('i,ij->ij',data['scale'],data['B'].T + data['B'])
         else: data = None
         #self.__dict__.update(create_shared_memory_dict(data))
         self.mat.update({key:create_shared_memory_dict(data)})
         self.set_model(self.mat[key]['model'][0])
        argv['mat'] = self.mat[0]
        #----------------------

        #Build elem_kappa_map
        self.elem_kappa_map = np.array([ list(self.mat[i]['kappa'])  for i in self.mesh['elem_mat_map']])
        #self.elem_kappa_map = np.array([ list(np.eye(2)*self.mat[i]['kappa'][0,0])  for i in self.mesh['elem_mat_map']])


        if comm.rank == 0 and self.verbose: self.bulk_info()

        if self.model[0:3] == 'mfp' or self.model[0:3] == 'rta' or self.model[0:3] == 'Gra':
         self.n_serial = self.mat[0]['tc'].shape[0]  
         self.n_parallel = self.mat[0]['tc'].shape[1]  
         block =  self.n_serial//comm.size
         self.ff = range(block*comm.rank,self.n_serial) if comm.rank == comm.size-1 else range(block*comm.rank,block*(comm.rank+1))
         if comm.rank == 0 and self.verbose: self.mfp_info()


        elif self.model == 'full': 
         self.n_parallel = len(self.mat[0]['tc'])
         if comm.rank == 0 and self.verbose: self.full_info() 
        #------------------------------------------

        if comm.rank == 0:
            if self.verbose: 
                print(colored(' -----------------------------------------------------------','green'),flush=True)
                print(" ",flush=True)

        if comm.rank == 0:
          data = self.solve_fourier(self.elem_kappa_map)

          
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
            data = argv['solve'](**argv)
            #data = solve_interface(**argv)

         elif self.model[0:3] == 'mfp' or \
            self.model[0:3] == 'Gra' :
            argv['n_serial'] = self.n_serial
            #a = time.time()
            #data = self.solve_mfp(**argv) 
            #quit()
            #a = time.time()
            #if argv.setdefault('deviational',False):
            # data = solve_deviational(**argv) 
            #else: 
            data = solve_mfp(**argv) 
            #print(time.time()-a)
         elif self.model[0:3] == 'rta':
            argv['n_serial'] = self.n_serial
            data = solve_rta(**argv) 
            #data = self.solve_rta(**argv)
         elif self.model == 'full':
            data = solve_full(**argv) 
            #data = self.solve_full(**argv)


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
          print(colored('  Deviational                              ','green')+ str(self.deviational),flush=True)
          print(colored('  Load State                               ','green')+ str(self.load_state),flush=True)
          print(colored('  Save State                               ','green')+ str(self.save_state),flush=True)
          print(colored('  Multiscale Error:                        ','green')+ str(self.error_multiscale),flush=True)
          print(colored('  Only Fourier:                            ','green')+ str(self.only_fourier),flush=True)
          print(colored('  Max Fourier Error:                       ','green')+ '%.1E' % (self.max_fourier_error),flush=True)
          print(colored('  Max Fourier Iter:                        ','green')+ str(self.max_fourier_iter),flush=True)
          print(colored('  Max BTE Error:                           ','green')+ '%.1E' % (self.max_bte_error),flush=True)
          print(colored('  Min BTE Iter:                            ','green')+ str(self.min_bte_iter),flush=True)
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



  def get_decomposed_directions(self,ll,rot):

    normal = self.mesh['face_normals'][ll,0:self.dim]
    dist   = self.mesh['dists'][ll,0:self.dim]
    v_orth = np.dot(normal,np.dot(rot,normal))/np.dot(normal,dist)
    v_non_orth = np.dot(rot,normal) - dist*v_orth

    return v_orth,v_non_orth[:self.dim]

  def get_kappa(self,i,j,ll,kappa):

   if i ==j:
    return np.array(kappa[i])
   
   normal = self.mesh['face_normals'][ll,0:self.dim]

   kappa_i = np.array(kappa[i])
   kappa_j = np.array(kappa[j])

   ki = np.dot(normal,np.dot(kappa_i,normal))
   kj = np.dot(normal,np.dot(kappa_j,normal))
   w  = self.mesh['interp_weigths'][ll]

   kappa_loc = kj*kappa_i/(ki*(1-w) + kj*w)

   return kappa_loc

  

  def compute_secondary_flux(self,temp,kappa):

   #if not np.isscalar(kappa):
   #  kappa = kappa[0,0]

   diff_temp = []
   for i in self.mesh['side_per_elem']:
       diff_temp.append(i*[0])

   a = time.time()
   for ll in self.mesh['active_sides'] :
    elems = self.mesh['side_elem_map_vec'][ll]
    kc1 = elems[0]
    c1 = self.mesh['centroids'][kc1]

    ind1 = list(self.mesh['elem_side_map_vec'][kc1]).index(ll)
    if not ll in self.mesh['boundary_sides']: 
     kc2 = elems[1]
     ind2 = list(self.mesh['elem_side_map_vec'][kc2]).index(ll)
     temp_1 = temp[kc1]
     temp_2 = temp[kc2]

     if ll in self.mesh['periodic_sides']:
      temp_2 += self.mesh['periodic_side_values'][list(self.mesh['periodic_sides']).index(ll)]

     diff_t = temp_2 - temp_1
     diff_temp[kc1][ind1]  = diff_t
     diff_temp[kc2][ind2]  = -diff_t

   gradT = np.zeros((self.n_elems,self.dim))
   for k,dt in enumerate(diff_temp):
    gradT[k] = np.einsum('js,s->j',self.mesh['weigths'][k,:,:self.mesh['side_per_elem'][k]],np.array(dt))

   #-----------------------------------------------------
   F_ave = np.zeros((len(self.mesh['sides']),self.dim))
   for ll in self.mesh['active_sides']:
      (i,j) = self.mesh['side_elem_map_vec'][ll]
      if not i==j:
       kappa_loc = self.get_kappa(i,j,ll,kappa)
       w = self.mesh['interp_weigths'][ll]
       F_ave[ll] = w*gradT[i] + (1.0-w)*gradT[j]
    

   C = np.zeros(self.n_elems)
   for ll in self.mesh['active_sides']:
    (i,j) = self.mesh['side_elem_map_vec'][ll]
    if not i==j:
      kappa_loc = self.get_kappa(i,j,ll,kappa)
      area = self.mesh['areas'][ll]   
      (dummy,v_non_orth) = self.get_decomposed_directions(ll,rot=kappa_loc)
      tmp = np.dot(F_ave[ll],v_non_orth)*area
      C[i] += tmp/self.mesh['volumes'][i]
      C[j] -= tmp/self.mesh['volumes'][j]


   return C,gradT


  def solve_fourier(self,kappa,**argv):

   if np.isscalar(kappa):
       kappa = np.diag(np.diag(kappa*np.eye(self.dim)))

   if kappa.ndim == 2:
      kappa = np.repeat(np.array([np.diag(np.diag(kappa))]),self.n_elems,axis=0)

   F = sp.dok_matrix((self.n_elems,self.n_elems))
   B = np.zeros(self.n_elems)
   for ll in self.mesh['active_sides']:

      area = self.mesh['areas'][ll] 
      (i,j) = self.mesh['side_elem_map_vec'][ll]
      vi = self.mesh['volumes'][i]
      vj = self.mesh['volumes'][j]
      kappa_loc = self.get_kappa(i,j,ll,kappa)
      #kappa_loc = np.eye(2)*kappa_loc[0,0] 

      if not i == j:

       (v_orth,dummy) = self.get_decomposed_directions(ll,rot=kappa_loc)
       F[i,i] += v_orth/vi*area
       F[i,j] -= v_orth/vi*area
       F[j,j] += v_orth/vj*area
       F[j,i] -= v_orth/vj*area
       if ll in self.mesh['periodic_sides']:    
        kk = list(self.mesh['periodic_sides']).index(ll)   
        B[i] += self.mesh['periodic_side_values'][kk]*v_orth/vi*area
        B[j] -= self.mesh['periodic_side_values'][kk]*v_orth/vj*area
   
    
   ##rescaleand fix one point to 0
   #scale = 1/F.max(axis=0).toarray()[0]
   #n = np.random.randint(self.n_elems)
   #scale[n] = 0
   #F.data = F.data * scale[F.indices]
   #F[n,n] = 1
   #B[n] = 0
   SU = splu(F.tocsc())
    #-----------------------

   C = np.zeros(self.n_elems)
   n_iter = 0
   kappa_old = 0
   error = 1  
   grad = np.zeros((self.n_elems,self.dim))
   while error > self.max_fourier_error and \
                  n_iter < self.max_fourier_iter :
        RHS = B + C
        #for n in range(self.n_elems):
        #  RHS[n] = RHS[n]*scale[n]  

        temp = SU.solve(RHS)
        temp = temp - (max(temp)+min(temp))/2.0
        kappa_eff = self.compute_diffusive_thermal_conductivity(temp,grad,kappa)
        error = abs((kappa_eff - kappa_old)/kappa_eff)
        kappa_old = kappa_eff
        n_iter +=1

        C,grad = self.compute_secondary_flux(temp,kappa)


   flux = -np.einsum('cij,cj->ci',kappa,grad)
 
   meta = [kappa_eff,error,n_iter] 
   return {'flux_fourier':flux,'temperature_fourier':temp,'meta':np.array(meta),'grad':grad}




  def compute_diffusive_thermal_conductivity(self,temp,gradT,kappa):

   kappa_eff = 0
   for l in self.mesh['flux_sides']:

    (i,j) = self.mesh['side_elem_map_vec'][l]

    
    if np.isscalar(kappa):
      kappa = np.diag(np.diag(kappa*np.eye(self.dim)))
 
    if kappa.ndim == 3:
      kappa = self.get_kappa(i,j,l,kappa)
      
    (v_orth,v_non_orth) = self.get_decomposed_directions(l,rot=kappa)

    deltaT = temp[i] - (temp[j] + 1)
    kappa_eff -= v_orth *  deltaT * self.mesh['areas'][l]
    w  = self.mesh['interp_weigths'][l]
    grad_ave = w*gradT[i] + (1.0-w)*gradT[j]

    kappa_eff += np.dot(grad_ave,v_non_orth)/2 * self.mesh['areas'][l]


   return kappa_eff*self.kappa_factor

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
    print(colored('  Contact:          ','green') + 'romanog@mit.edu                       ',flush=True) 
    print(colored('  Source code:      ','green') + 'https://github.com/romanodev/OpenBTE  ',flush=True)
    print(colored('  Become a sponsor: ','green') + 'https://github.com/sponsors/romanodev ',flush=True)
    print(colored('  Cloud:            ','green') + 'https://shorturl.at/cwDIP             ',flush=True)
    print(colored('  Mailing List:     ','green') + 'https://shorturl.at/admB0             ',flush=True)
    print(colored(' -----------------------------------------------------------','green'),flush=True)
    print(flush=True)   

