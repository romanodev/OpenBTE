from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
import deepdish as dd
from mpi4py import MPI
import scipy.sparse as sp
import time

comm = MPI.COMM_WORLD


class Solver(object):

  def __init__(self,**argv):
        self.data = argv
        self.tt = np.float64
        self.state = {}

 
        #-----IMPORT MESH-------------------
        if 'geometry' in argv.keys():
         self.mesh = argv['geometry'].data
        else: 
         #self.mesh = load_dictionary('geometry.h5')
         self.mesh = dd.io.load('geometry.h5')
  
        self.n_elems = self.mesh['n_elems'][0]
        self.im = np.concatenate((self.mesh['i'],list(np.arange(self.n_elems))))
        self.jm = np.concatenate((self.mesh['j'],list(np.arange(self.n_elems))))

        #---------------------------------------
        self.verbose = argv.setdefault('verbose',True)
        self.alpha = argv.setdefault('alpha',1.0)
        self.n_side_per_elem = self.mesh['n_side_per_elem'][0]

        #-----IMPORT MATERIAL-------------------
        self.mat = dd.io.load('material.h5')

        self.kappa = self.mat['kappa']
        self.kappa_vec = np.zeros((len(self.mesh['elems']),3,3))

        for n in self.mesh['elem_mat_map'].keys():
            if self.mesh['elem_mat_map'][n] == 0:
                self.kappa_vec[n] = 0.5* np.eye(3)
            else:    
                self.kappa_vec[n] =  np.eye(3)

        for i in range(len(self.mesh['elems'])):
          self.kappa_vec[i] = self.kappa*np.eye(3)
         
        self.only_fourier = argv.setdefault('only_fourier',False)
        if not self.only_fourier:
         self.tc = self.mat['temp']
         self.n_index = len(self.tc)
         #Get collision matrix------------------------------------
         if len(self.mat['B']) == 0:
           self.coll = False
           self.ac = self.mat['ac']
         else:  
           self.coll = True
           if comm.rank == 0:
            B = self.mat['B']
            B += B.T - 0.5*np.diag(np.diag(B))
            B = np.einsum('i,ij->ij',self.mat['scale'],B)
           else : B = None
           self.BM = comm.bcast(B,root=0) 
         #-------------------------------------------------------
 
         self.im = np.concatenate((self.mesh['i'],list(np.arange(self.n_elems))))
         self.jm = np.concatenate((self.mesh['j'],list(np.arange(self.n_elems))))
         self.sigma = self.mat['G']*1e9
         self.VMFP = self.mat['F']*1e9
         self.n_index = len(self.tc)
         #MPI info
         block =  self.n_index//comm.size
         if comm.rank == comm.size-1: 
          self.rr = range(block*comm.rank,self.n_index)
         else: 
          self.rr = range(block*comm.rank,block*(comm.rank+1))
        #------------------
         #----------------IMORT MATERIAL-------
  

        self.kappa_factor = self.mesh['kappa_factor'][0]
        if comm.rank == 0:
         if self.verbose: self.print_logo()
  
         self.assemble_fourier(argv)
 
         if self.verbose:
          print('                        SYSTEM INFO                 ')   
          print(colored(' -----------------------------------------------------------','green'))
          print(colored('  Space Discretization:                    ','green') + str(self.n_elems))
          if not self.only_fourier:
           print(colored('  Momentum Discretization:                 ','green') + str(len(self.tc)))
          print(colored('  Bulk Thermal Conductivity [W/m/K]:       ','green')+ str(round(self.mat['kappa'][0,0],4)))
          #solve fourier
          data = self.solve_fourier(**argv)
        else : data = None
        data = comm.bcast(data,root=0)


        variables = {0:{'name':'Temperature Fourier','units':'K',        'data':data['temperature']},\
                     1:{'name':'Flux Fourier'       ,'units':'W/m/m/K','data':data['flux']}}
 
        self.state.update({'variables':variables,\
                           'kappa_fourier':data['kappa']})

        if comm.rank == 0:
         if self.verbose:
          print(colored('  Fourier Thermal Conductivity [W/m/K]:    ','green') + str(round(self.state['kappa_fourier'][0],4)))
          print(colored(' -----------------------------------------------------------','green'))
  
          if not self.only_fourier:
           print()
           print('      Iter    Thermal Conductivity [W/m/K]      Error ''')
           print(colored(' -----------------------------------------------------------','green'))

        if not self.only_fourier:
         data = self.solve_bte(**argv)

         variables = self.state['variables']

         variables[2]    = {'name':'Temperature BTE','units':'K'             ,'data':data['temperature']}
         variables[3]    = {'name':'Flux BTE'       ,'units':'W/m/m/K'       ,'data':data['flux']}

         self.state.update({'variables':variables,\
                           'kappa':data['kappa_vec']})


        if comm.rank == 0:
         if argv.setdefault('save',True):
          dd.io.save('solver.h5',self.state)
         if self.verbose:
          print(' ')   
          print(colored('                 OpenBTE ended successfully','green'))
          print(' ')  




  def solve_bte(self,**argv):


     if comm.rank == 0:
      SS = np.zeros(1)
      Gbp = np.zeros(1)
      if len(self.mesh['db']) > 0:
       Gb = np.einsum('qj,jn->qn',self.VMFP,self.mesh['db'],optimize=True)
       Gbp = Gb.clip(min=0);Gbm2 = Gb.clip(max=0)
       Gb = np.einsum('qj,jn->qn',self.sigma,self.mesh['db'],optimize=True)
       Gbp = Gb.clip(min=0); Gbm = Gb.clip(max=0)
       SS = np.einsum('qc,c->qc',Gbm2,1/Gbm.sum(axis=0))
    
      #---------------------------------------------------------------
      BB = -np.outer(self.sigma[:,0],np.sum(self.mesh['B_with_area_old'],axis=0))*self.kappa_factor*1e-18

      data = {'BB':BB,'SS':SS,'Gbp':Gbp}
     else: data = None 
     data = comm.bcast(data,root = 0)

     #Main matrix----
     G = np.einsum('qj,jn->qn',self.VMFP[self.rr],self.mesh['k'],optimize=True)
     Gp = G.clip(min=0); Gm = G.clip(max=0)
     D = np.ones((len(self.rr),self.n_elems))
     for n,i in enumerate(self.mesh['i']): D[:,i] += Gp[:,n]
     if len(self.mesh['db']) > 0:
      Gb = np.einsum('qj,jn->qn',self.VMFP[self.rr],self.mesh['db'],optimize=True)
      Gbp = Gb.clip(min=0);
      for n,(i,j) in enumerate(zip(self.mesh['eb'],self.mesh['sb'])):
         D[:,i]  += Gbp[:,n]
     A = np.concatenate((Gm,D),axis=1)
     P = np.zeros((len(self.rr),self.n_elems))
     for n,(i,j) in enumerate(zip(self.mesh['i'],self.mesh['j'])): P[:,i] -= Gm[:,n]*self.mesh['B'][i,j]

     lu = {i:sp.linalg.splu(sp.csc_matrix((A[n],(self.im,self.jm)),shape=(self.n_elems,self.n_elems),dtype=self.tt)   ) for n,i in enumerate(self.rr) }

     #Periodic------------------
     #Compute boundary-----------------------------------------------
     X = np.tile(self.state['variables'][0]['data'],(self.n_index,1))
     X_old = X.copy()
     kappa_vec = [self.state['kappa_fourier'][0]]
     kappa_old = kappa_vec[-1]
     alpha = self.data.setdefault('alpha',1)
     error = 1
     kk = 0

     Xp = np.zeros_like(X)
     Bm = np.zeros((self.n_index,self.n_elems))
     DeltaTp = np.zeros(self.n_elems)   
     DeltaT = np.zeros(self.n_elems)   

     kappa = np.zeros(1)
     while kk < self.data.setdefault('max_bte_iter',100) and error > self.data.setdefault('max_bte_error',1e-2):

      #Boundary-----   
      a = time.time()
      Bm = np.zeros((len(self.rr),self.n_elems))   
      if len(self.mesh['db']) > 0:
       for n,i in enumerate(self.mesh['eb']): Bm[:,i] +=np.einsum('u,u,q->q',X[:,i],data['Gbp'][:,n],data['SS'][self.rr,n],optimize=True)
      #---------------------

      if self.coll:
       DeltaT = np.matmul(self.BM[self.rr],alpha*X+(1-alpha)*X_old) 
       for n,i in enumerate(self.rr): Xp[i] = lu[i].solve(DeltaT[n] + P[n] + Bm[n]) 
      else: 
       DeltaTp = np.dot(X[self.rr].T, self.ac[self.rr])
       comm.Allreduce([DeltaTp,MPI.DOUBLE],[DeltaT,MPI.DOUBLE],op=MPI.SUM)

       for n,i in enumerate(self.rr): Xp[i] = lu[i].solve(DeltaT + P[n] + Bm[n]) 

      kappap = np.array([np.sum(np.multiply(data['BB'][self.rr],Xp[self.rr] ))])

      comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
      comm.Allreduce([Xp,MPI.DOUBLE],[X,MPI.DOUBLE],op=MPI.SUM)

      kk +=1

      error = abs(kappa_old-kappa[0])/abs(kappa[0])
      kappa_old = kappa[0]
      kappa_vec.append(kappa[0])
      if self.verbose and comm.rank == 0:   
       print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa_vec[-1],error))

     if self.verbose and comm.rank == 0:
      print(colored(' -----------------------------------------------------------','green'))


     T = np.einsum('qc,q->c',X,self.tc)
     J = np.einsum('qj,qc->cj',self.sigma,X)*1e-18
     return {'kappa_vec':kappa_vec,'temperature':T,'flux':J}



  def get_decomposed_directions(self,i,j,rot=np.eye(3)):

     normal = self.mesh['normals'][i][j]
     dist   = self.mesh['dists'][i][j]
     v_orth = np.dot(normal,np.dot(rot,normal))/np.dot(normal,dist)
     v_non_orth = np.dot(rot,normal) - dist*v_orth
     return v_orth,v_non_orth

  def get_kappa(self,i,j,ll):

   if i ==j:
    return np.array(self.kappa_vec[i])
   

   normal = self.mesh['normals'][i][j]

   kappa_i = np.array(self.kappa_vec[i])
   kappa_j = np.array(self.kappa_vec[j])

   ki = np.dot(normal,np.dot(kappa_i,normal))
   kj = np.dot(normal,np.dot(kappa_j,normal))
   w  = self.mesh['interp_weigths'][ll][0]

   kappa = kj*kappa_i/(ki*(1-w) + kj*w)
 
   return kappa

    
  def assemble_fourier(self,data):

    F = sp.dok_matrix((self.n_elems,self.n_elems))
    B = np.zeros(self.n_elems)

    for ll in self.mesh['side_list']['active']:
      area = self.mesh['areas'][ll]  
      (i,j) = self.mesh['side_elem_map'][ll]
      vi = self.mesh['volumes'][i]
      vj = self.mesh['volumes'][j]
      kappa = self.get_kappa(i,j,ll)
      if not i == j:
       (v_orth,dummy) = self.get_decomposed_directions(i,j,rot=kappa)
       F[i,i] += v_orth/vi*area
       F[i,j] -= v_orth/vi*area
       F[j,j] += v_orth/vj*area
       F[j,i] -= v_orth/vj*area
       if ll in self.mesh['side_list']['Periodic']:
        #B[i] += self.mesh['periodic_values'][i][j][0]*v_orth/vi*area
        #B[j] += self.mesh['periodic_values'][j][i][0]*v_orth/vj*area
        B[i] += self.mesh['periodic_side_values'][ll]*v_orth/vi*area
        B[j] -= self.mesh['periodic_side_values'][ll]*v_orth/vj*area
    self.fourier = {'Af':F.tocsc(),'Bf':B}



  def solve_fourier(self,**argv):

    SU = splu(self.fourier['Af'])

    C = np.zeros(self.n_elems)

    n_iter = 0
    kappa_old = 0
    error = 1  
    grad = np.zeros((self.n_elems,3))
    while error > argv.setdefault('max_fourier_error',1e-4) and \
                  n_iter < argv.setdefault('max_fourier_iter',10) :
    
        RHS = self.fourier['Bf'] + C
        temp = SU.solve(RHS)
        temp = temp - (max(temp)+min(temp))/2.0
        kappa_eff = self.compute_diffusive_thermal_conductivity(temp,grad)
        error = abs((kappa_eff - kappa_old)/kappa_eff)
        kappa_old = kappa_eff
        n_iter +=1
        grad = self.compute_grad(temp)
        
        C = self.compute_non_orth_contribution(grad)

    flux = -np.einsum('cij,cj->ci',self.kappa_vec,grad)
    return {'flux':flux,'temperature':temp,'kappa':np.array([kappa_eff])}



  def compute_grad(self,temp):

   diff_temp = self.n_elems*[None]
   for i in range(len(diff_temp)):
      diff_temp[i] = len(self.mesh['elems'][i])*[0] 
   

   gradT = np.zeros((self.n_elems,3))
   for ll in self.mesh['side_list']['active'] :
    elems = self.mesh['side_elem_map'][ll]

    kc1 = elems[0]
    c1 = self.mesh['centroids'][kc1]

    ind1 = list(self.mesh['elem_side_map'][kc1]).index(ll)

    if not ll in self.mesh['side_list']['Boundary']:

     kc2 = elems[1]
     ind2 = list(self.mesh['elem_side_map'][kc2]).index(ll)
     temp_1 = temp[kc1]
     temp_2 = temp[kc2]

     if ll in self.mesh['side_list']['Periodic']:
      #temp_2 -= self.mesh['periodic_values'][kc2][kc1][0]
      temp_2 -= self.mesh['periodic_side_values'][ll]

     diff_t = temp_2 - temp_1
     
     diff_temp[kc1][ind1]  = diff_t
     diff_temp[kc2][ind2]  = -diff_t

   
   for k in range(self.n_elems) :
    tmp = np.dot(self.mesh['weigths'][k],diff_temp[k])
    gradT[k,0] = tmp[0] #THESE HAS TO BE POSITIVE
    gradT[k,1] = tmp[1]
    if self.mesh['dim'] == 3:
     gradT[k,2] = tmp[2]

   return gradT  


  def compute_non_orth_contribution(self,gradT) :

    C = np.zeros(self.n_elems)

    for ll in self.mesh['side_list']['active']:

     (i,j) = self.mesh['side_elem_map'][ll]

     if not i==j:

      area = self.mesh['areas'][ll]   
      w  = self.mesh['interp_weigths'][ll][0]
      #F_ave = w*np.dot(gradT[i],self.mat['kappa']) + (1.0-w)*np.dot(gradT[j],self.mat['kappa'])
      F_ave = w*np.dot(gradT[i],self.kappa_vec[i]) + (1.0-w)*np.dot(gradT[j],self.kappa_vec[j])
      grad_ave = w*gradT[i] + (1.0-w)*gradT[j]

      (_,v_non_orth) = self.get_decomposed_directions(i,j)#,rot=self.mat['kappa'])

      C[i] += np.dot(F_ave,v_non_orth)/2.0/self.mesh['volumes'][i]*area
      C[j] -= np.dot(F_ave,v_non_orth)/2.0/self.mesh['volumes'][j]*area

    return C


  def compute_diffusive_thermal_conductivity(self,temp,gradT):

   kappa = 0
   for l in self.mesh['flux_sides']:

    (i,j) = self.mesh['side_elem_map'][l]
    #(v_orth,v_non_orth) = self.get_decomposed_directions(i,j,rot=self.mat['kappa'])
    (v_orth,v_non_orth) = self.get_decomposed_directions(i,j,rot=self.get_kappa(i,j,l))

    deltaT = temp[i] - (temp[j] + 1) 
    kappa -= v_orth *  deltaT * self.mesh['areas'][l]
    w  = self.mesh['interp_weigths'][l][0]
    grad_ave = w*gradT[i] + (1.0-w)*gradT[j]
    kappa += np.dot(grad_ave,v_non_orth)/2 * self.mesh['areas'][l]

   return kappa*self.kappa_factor

  def print_logo(self):


    #v = pkg_resources.require("OpenBTE")[0].version   
    print(' ')
    print(colored(r'''        ___                   ____ _____ _____ ''','green'))
    print(colored(r'''       / _ \ _ __   ___ _ __ | __ )_   _| ____|''','green'))
    print(colored(r'''      | | | | '_ \ / _ \ '_ \|  _ \ | | |  _|  ''','green'))
    print(colored(r'''      | |_| | |_) |  __/ | | | |_) || | | |___ ''','green'))
    print(colored(r'''       \___/| .__/ \___|_| |_|____/ |_| |_____|''','green'))
    print(colored(r'''            |_|                                ''','green'))
    print()
    print('                       GENERAL INFO')
    print(colored(' -----------------------------------------------------------','green'))
    print(colored('  Contact:          ','green') + 'romanog@mit.edu                       ') 
    print(colored('  Source code:      ','green') + 'https://github.com/romanodev/OpenBTE  ')
    print(colored('  Become a sponsor: ','green') + 'https://github.com/sponsors/romanodev ')
    print(colored('  Cloud:            ','green') + 'https://shorturl.at/cwDIP             ')
    print(colored('  Mailing List:     ','green') + 'https://shorturl.at/admB0             ')
    print(colored(' -----------------------------------------------------------','green'))
    print()   

