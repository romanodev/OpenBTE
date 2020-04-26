from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
import deepdish as dd
from mpi4py import MPI
import scipy.sparse as sp
import plotly.express as px
import plotly.graph_objects as go
import time


comm = MPI.COMM_WORLD


class Solver(object):

  def __init__(self,**argv):

        self.data = argv
        self.tt = np.float64
        self.state = {}
        self.multiscale = argv.setdefault('multiscale',False)

        #-----IMPORT MESH-------------------
        if 'geometry' in argv.keys():
         self.mesh = argv['geometry'].data
        else: 
         self.mesh = dd.io.load('geometry.h5')

        self.n_elems = self.mesh['n_elems'][0]
        self.im = np.concatenate((self.mesh['i'],list(np.arange(self.n_elems))))
        self.jm = np.concatenate((self.mesh['j'],list(np.arange(self.n_elems))))
        self.ne = self.mesh['n_elems'][0]

        #---------------------------------------
        self.verbose = argv.setdefault('verbose',True)
        self.alpha = argv.setdefault('alpha',1.0)
        self.n_side_per_elem = self.mesh['n_side_per_elem'][0]

        #-----IMPORT MATERIAL-------------------
        self.mat = dd.io.load('material.h5')

        self.kappa = self.mat['kappa']

        self.only_fourier = argv.setdefault('only_fourier',False)
        if not self.only_fourier:
         self.tc = self.mat['temp']
         if self.tc.ndim == 1:
            self.coll = True
            self.n_serial = 1
            self.n_parallel = len(self.tc)
            self.tc = np.array([self.tc])
         else:   
            self.n_serial = self.tc.shape[0]
            self.n_parallel = self.tc.shape[1]
             
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
         if self.coll : self.sigma = np.array([self.sigma])
         self.VMFP = self.mat['F']*1e9
         if self.coll : self.VMFP = np.array([self.VMFP])
         #self.n_index = np.shape(self.tc)[1]
         #MPI info
         block =  self.n_parallel//comm.size
         if comm.rank == comm.size-1: 
          self.rr = range(block*comm.rank,self.n_parallel)
         else: 
          self.rr = range(block*comm.rank,block*(comm.rank+1))
   
         #--------------------------------     
         block =  self.n_serial//comm.size
         if comm.rank == comm.size-1: 
          self.ff = range(block*comm.rank,self.n_serial)
         else: 
          self.ff = range(block*comm.rank,block*(comm.rank+1))
         #--------------------------------
  
        self.kappa_factor = self.mesh['kappa_factor'][0]
        if comm.rank == 0:
         if self.verbose: self.print_logo()
  
         #self.assemble_fourier()
         #self.assemble_modified_fourier()
 
         if self.verbose:
          print('                        SYSTEM INFO                 ')   
          print(colored(' -----------------------------------------------------------','green'))
          print(colored('  Space Discretization:                    ','green') + str(self.n_elems))
          if not self.only_fourier:
            print(colored('  Momentum Discretization:                 ','green') + str(np.prod(self.tc.shape)))
          print(colored('  Bulk Thermal Conductivity [W/m/K]:       ','green')+ str(round(self.mat['kappa'][0,0],4)))
          #solve fourier
          data = self.solve_fourier_new(self.kappa,**argv)

        else : data = None
        data = comm.bcast(data,root=0)
 

        variables = {0:{'name':'Temperature Fourier','units':'K',        'data':data['temperature']},\
                     1:{'name':'Flux Fourier'       ,'units':'W/m/m/K','data':data['flux']}}
 
        self.state.update({'variables':variables,\
                           'kappa_fourier':data['kappa']})

        if comm.rank == 0:
         if self.verbose:
          print(colored('  Fourier Thermal Conductivity [W/m/K]:    ','green') + str(round(self.state['kappa_fourier'],4)))
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
       Gb = np.einsum('mqj,jn->mqn',self.VMFP,self.mesh['db'],optimize=True)
       Gbp = Gb.clip(min=0); Gbm2 = Gb.clip(max=0)
       Gb = np.einsum('mqj,jn->mqn',self.sigma,self.mesh['db'],optimize=True)
       Gbp = Gb.clip(min=0); Gbm = Gb.clip(max=0)

       if self.coll:   
        SS  = np.einsum('mqc,c->mqc',Gbm2,1/Gbm.sum(axis=0).sum(axis=0))
       else: 
        with np.errstate(divide='ignore', invalid='ignore'):
         tmp = 1/Gbm.sum(axis=1)
         tmp[np.isinf(tmp)] = 0
        SS = np.einsum('mqc,mc->mqc',Gbm2,tmp)
      #---------------------------------------------------------------
      data1 = {'Gbp':Gbp}
      data2 = {'SS':SS}
     else: data1 = None; data2 = None 
     data1 = comm.bcast(data1,root = 0)
     data2 = comm.bcast(data2,root = 0)
     Gbp = data1['Gbp']
     SS = data2['SS']

     #Main matrix----
     G = np.einsum('mqj,jn->mqn',self.VMFP[:,self.rr],self.mesh['k'],optimize=True)
     Gp = G.clip(min=0); Gm = G.clip(max=0)

     D = np.ones((self.n_serial,len(self.rr),self.ne))
     for n,i in enumerate(self.mesh['i']): 
         D[:,:,i] += Gp[:,:,n]
     if len(self.mesh['db']) > 0:
      Gb = np.einsum('mqj,jn->mqn',self.VMFP[:,self.rr],self.mesh['db'],optimize=True)
      Gbp2 = Gb.clip(min=0);
      for n,i in enumerate(self.mesh['eb']): D[:,:,i]  += Gbp2[:,:,n]

     A = np.concatenate((Gm,D),axis=2)
     P = np.zeros((self.n_serial,len(self.rr),self.n_elems))
     for n,(i,j) in enumerate(zip(self.mesh['i'],self.mesh['j'])): 
         P[:,:,i] -= Gm[:,:,n]*self.mesh['B'][i,j]

     lu =  {}

     X = np.tile(self.state['variables'][0]['data'],(self.n_serial,self.n_parallel,1))
     X_old = X.copy()
     kappa_vec = [self.state['kappa_fourier']]
     kappa_old = kappa_vec[-1]
     alpha = self.data.setdefault('alpha',1)
     error = 1
     kk = 0

     Xp = np.zeros_like(X)
     Bm = np.zeros((self.n_parallel,self.n_elems))
     DeltaT = np.zeros(self.n_elems)   

     kappa_tot = np.zeros(1)
     MM = np.zeros(1)
     Mp = np.zeros(1)
     kappap = np.zeros((self.n_serial,self.n_parallel))
     kappa = np.zeros((self.n_serial,self.n_parallel))
     if not self.coll:
      tf = np.zeros((self.n_serial,self.mesh['n_elems'][0]))
      tfg = np.zeros((self.n_serial,self.mesh['n_elems'][0],3))
      tfp = np.zeros((self.n_serial,self.mesh['n_elems'][0]))
      tfgp = np.zeros((self.n_serial,self.mesh['n_elems'][0],3))
     termination = True
     if self.multiscale:
      mfp_ave = np.sqrt(3*self.mat['mfp_average'])*1e9
     kappafp = np.zeros((self.n_serial,self.n_parallel));kappaf = np.zeros((self.n_serial,self.n_parallel))
     while kk < self.data.setdefault('max_bte_iter',100) and error > self.data.setdefault('max_bte_error',1e-2):

      # COMMON----- 
       Bm = np.zeros((self.n_serial,len(self.rr),self.n_elems))   
       if len(self.mesh['db']) > 0: 
         for n,i in enumerate(self.mesh['eb']):
              if self.coll:
               Bm[:,:,i] +=np.einsum('mu,mu,lq->lq',X[:,:,i],Gbp[:,:,n],SS[:,self.rr,n],optimize=True)
              else: 
               Bm[:,:,i] +=np.einsum('mu,mu,mq->mq',X[:,:,i],Gbp[:,:,n],SS[:,self.rr,n],optimize=True)
       
       if self.coll:
        DeltaT = np.matmul(self.BM[self.rr],alpha*X[0]+(1-alpha)*X_old[0]) 
        for n,i in enumerate(self.rr):

            if not i in lu.keys() :
                lu_loc = sp.linalg.splu(sp.csc_matrix((A[0,n],(self.im,self.jm)),shape=(self.ne,self.ne),dtype=self.tt))
                if argv.setdefault('keep_lu',True):
                 lu.update({i:lu_loc})
            else: lu_loc   = lu[i]

            Xp[0,i] = lu_loc.solve(DeltaT[n] + Bm[0,n] + P[0,n])
            kappap[0,i] =  -np.dot(self.mesh['kappa_mask'],Xp[0,i])
        comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Xp,MPI.DOUBLE],[X,MPI.DOUBLE],op=MPI.SUM)
       else: 

        DeltaTp = np.einsum('mqc,mq->c',X[:,self.rr], self.ac[:,self.rr])
        comm.Allreduce([DeltaTp,MPI.DOUBLE],[DeltaT,MPI.DOUBLE],op=MPI.SUM)

        if self.multiscale:
         for n,m in enumerate(self.ff):
           dataf = self.solve_fourier_new(self.mat['mfp_average'][m]*1e-18,**argv,pseudo=DeltaT)
           tfp[m] = dataf['temperature']
           tfgp[m] = dataf['grad']
           for q in range(self.n_parallel): 
            kappafp[m,q] = -np.dot(self.mesh['kappa_mask'],dataf['temperature'] - np.dot(self.VMFP[m,q],dataf['grad'].T))
         comm.Allreduce([kappafp,MPI.DOUBLE],[kappaf,MPI.DOUBLE],op=MPI.SUM)
         comm.Allreduce([tfp,MPI.DOUBLE],[tf,MPI.DOUBLE],op=MPI.SUM)
         comm.Allreduce([tfgp,MPI.DOUBLE],[tfg,MPI.DOUBLE],op=MPI.SUM)

        #Multiscale scheme-----------------------------
        diffusive = 0
        for n,q in enumerate(self.rr): 
           fourier = False
           for m in range(self.n_serial)[::-1]:
              if fourier:
               kappap[m,q] = kappaf[m,q]
               Xp[m,q] = tf[m] - np.dot(self.VMFP[m,q],tfg[m].T)
               diffusive +=1

              else: 
               if not (m,q) in lu.keys() :
                lu_loc = sp.linalg.splu(sp.csc_matrix((A[m,n],(self.im,self.jm)),shape=(self.ne,self.ne),dtype=self.tt))
                if argv.setdefault('keep_lu',True):
                 lu.update({(m,q):lu_loc})
               else: lu_loc   = lu[(m,q)]

               Xp[m,q] = lu_loc.solve(DeltaT + P[m,n] + Bm[m,n]) 
               kappap[m,q] = -np.dot(self.mesh['kappa_mask'],Xp[m,q])

               #if q == 20:
               #  print(abs(kappap[m,q] - kappaf[m,q])/abs(kappap[m,q]))  
                
                   
               if abs(kappap[m,q] - kappaf[m,q])/abs(kappap[m,q]) < 0.015 and self.multiscale:
                  kappap[m,q] = kappaf[m,q]
                  diffusive +=1
                  fourier=True
               else:   
                   if self.multiscale and m == 0:
                       termination = False
        Mp[0] = diffusive
        comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Xp,MPI.DOUBLE],[X,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Mp,MPI.DOUBLE],[MM,MPI.DOUBLE],op=MPI.SUM)

       #-----------------------------------------------
       kappa_totp = np.array([np.einsum('mq,mq->',self.sigma[:,self.rr,0],kappa[:,self.rr])])*self.kappa_factor*1e-18
       comm.Allreduce([kappa_totp,MPI.DOUBLE],[kappa_tot,MPI.DOUBLE],op=MPI.SUM)
       kk +=1

       error = abs(kappa_old-kappa_tot[0])/abs(kappa_tot[0])
       kappa_old = kappa_tot[0]
       kappa_vec.append(kappa_tot[0])
       if self.verbose and comm.rank == 0:   
        print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa_vec[-1],error))

     if self.verbose and comm.rank == 0:
      print(colored(' -----------------------------------------------------------','green'))


     if self.multiscale and comm.rank == 0:
        print()
        print('                  Multiscale Diagnostics        ''')
        print(colored(' -----------------------------------------------------------','green'))

        diff = int(MM[0])/self.n_serial/self.n_parallel 
        print(colored(' BTE:              ','green') + str(round((1-diff)*100,2)) + ' %' )
        print(colored(' FOURIER:          ','green') + str(round(diff*100,2)) + ' %' )
        print(colored(' Full termination: ','green') + str(termination) )
        print(colored(' -----------------------------------------------------------','green'))

     
     T = np.einsum('mqc,mq->c',X,self.tc)
     J = np.einsum('mqj,mqc->cj',self.sigma,X)*1e-18
     return {'kappa_vec':kappa_vec,'temperature':T,'flux':J}



  def get_decomposed_directions(self,i,j,rot=np.eye(3)):

     normal = self.mesh['normals'][i][j]
     dist   = self.mesh['dists'][i][j]
     v_orth = np.dot(normal,np.dot(rot,normal))/np.dot(normal,dist)
     v_non_orth = np.dot(rot,normal) - dist*v_orth
     return v_orth,v_non_orth

  def get_kappa(self,i,j,ll,kappa):

   if i ==j:
    return np.array(kappa[i])
   
   normal = self.mesh['normals'][i][j]

   kappa_i = np.array(kappa[i])
   kappa_j = np.array(kappa[j])

   ki = np.dot(normal,np.dot(kappa_i,normal))
   kj = np.dot(normal,np.dot(kappa_j,normal))
   w  = self.mesh['interp_weigths'][ll][0]

   kappa_loc = kj*kappa_i/(ki*(1-w) + kj*w)
 
   return kappa_loc

   
  def assemble_modified_fourier(self):

    F = sp.dok_matrix((self.n_elems,self.n_elems))
    B = np.zeros(self.n_elems)
    for ll in self.mesh['side_list']['active']:
      area = self.mesh['areas'][ll]  
      (i,j) = self.mesh['side_elem_map'][ll]
      vi = self.mesh['volumes'][i]
      vj = self.mesh['volumes'][j]
      kappa = self.get_kappa(i,j,ll)
      if not i == j:
       (v_orth,dummy) = self.get_decomposed_directions(i,j)
       F[i,i] += v_orth/vi*area
       F[i,j] -= v_orth/vi*area
       F[j,j] += v_orth/vj*area
       F[j,i] -= v_orth/vj*area
       if ll in self.mesh['side_list']['Periodic']:
        B[i] += self.mesh['periodic_side_values'][ll]*v_orth/vi*area
        B[j] -= self.mesh['periodic_side_values'][ll]*v_orth/vj*area
    self.mfe = {'Af':F.tocsc(),'Bf':B}



  def solve_fourier_new(self,kappa,**argv):

    if np.isscalar(kappa):
       kappa = np.diag(np.diag(kappa*np.eye(3)))

    if kappa.ndim == 2:
      kappa = np.repeat(np.array([np.diag(np.diag(kappa))]),self.mesh['n_elems'],axis=0)

    F = sp.dok_matrix((self.n_elems,self.n_elems))
    B = np.zeros(self.n_elems)
 
    for ll in self.mesh['side_list']['active']:
      area = self.mesh['areas'][ll]  
      (i,j) = self.mesh['side_elem_map'][ll]
      vi = self.mesh['volumes'][i]
      vj = self.mesh['volumes'][j]
      kappa_loc = self.get_kappa(i,j,ll,kappa)
      if not i == j:
       (v_orth,dummy) = self.get_decomposed_directions(i,j,rot=kappa_loc)
       F[i,i] += v_orth/vi*area
       F[i,j] -= v_orth/vi*area
       F[j,j] += v_orth/vj*area
       F[j,i] -= v_orth/vj*area
       if ll in self.mesh['side_list']['Periodic']:
        B[i] += self.mesh['periodic_side_values'][ll]*v_orth/vi*area
        B[j] -= self.mesh['periodic_side_values'][ll]*v_orth/vj*area
    #rescaleand fix one point to 0
    F = F.tocsc()
    if 'pseudo' in argv.keys():
      F = F + sp.eye(self.mesh['n_elems'])
      B = B + argv['pseudo']
      scale = 1/F.max(axis=0).toarray()[0]
      F.data = F.data * scale[F.indices]
    else:  
      scale = 1/F.max(axis=0).toarray()[0]
      n = np.random.randint(self.mesh['n_elems'])
      scale[n] = 0
      F.data = F.data * scale[F.indices]
      F[n,n] = 1
      B[n] = 0
    #-----------------------

    SU = splu(F)

    C = np.zeros(self.n_elems)
    
    n_iter = 0
    kappa_old = 0
    error = 1  
    grad = np.zeros((self.n_elems,3))
    while error > argv.setdefault('max_fourier_error',1e-4) and \
                  n_iter < argv.setdefault('max_fourier_iter',10) :
        RHS = B + C
        for n in range(self.mesh['n_elems'][0]):
          RHS[n] = RHS[n]*scale[n]  

        temp = SU.solve(RHS)
        temp = temp - (max(temp)+min(temp))/2.0
        kappa_eff = self.compute_diffusive_thermal_conductivity(temp,grad,kappa)
        error = abs((kappa_eff - kappa_old)/kappa_eff)
        kappa_old = kappa_eff
        n_iter +=1
        grad = self.compute_grad(temp)
        C = self.compute_non_orth_contribution(grad,kappa)
    flux = -np.einsum('cij,cj->ci',kappa,grad)

    return {'flux':flux,'temperature':temp,'kappa':kappa_eff,'grad':grad}

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
      temp_2 += self.mesh['periodic_side_values'][ll]

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


  def compute_non_orth_contribution(self,gradT,kappa) :

    C = np.zeros(self.n_elems)

    for ll in self.mesh['side_list']['active']:

     (i,j) = self.mesh['side_elem_map'][ll]

     if not i==j:

      area = self.mesh['areas'][ll]   
      w  = self.mesh['interp_weigths'][ll][0]
      #F_ave = w*np.dot(gradT[i],self.mat['kappa']) + (1.0-w)*np.dot(gradT[j],self.mat['kappa'])
      F_ave = w*np.dot(gradT[i],kappa[i]) + (1.0-w)*np.dot(gradT[j],kappa[j])
      grad_ave = w*gradT[i] + (1.0-w)*gradT[j]

      (_,v_non_orth) = self.get_decomposed_directions(i,j)#,rot=self.mat['kappa'])

      C[i] += np.dot(F_ave,v_non_orth)/2.0/self.mesh['volumes'][i]*area
      C[j] -= np.dot(F_ave,v_non_orth)/2.0/self.mesh['volumes'][j]*area

    return C


  def compute_diffusive_thermal_conductivity(self,temp,gradT,kappa):

   kappa_eff = 0
   for l in self.mesh['flux_sides']:

    (i,j) = self.mesh['side_elem_map'][l]
    #(v_orth,v_non_orth) = self.get_decomposed_directions(i,j,rot=self.mat['kappa'])
    (v_orth,v_non_orth) = self.get_decomposed_directions(i,j,rot=self.get_kappa(i,j,l,kappa))

    deltaT = temp[i] - (temp[j] + 1) 
    kappa_eff -= v_orth *  deltaT * self.mesh['areas'][l]
    w  = self.mesh['interp_weigths'][l][0]
    grad_ave = w*gradT[i] + (1.0-w)*gradT[j]
    kappa_eff += np.dot(grad_ave,v_non_orth)/2 * self.mesh['areas'][l]

   return kappa_eff*self.kappa_factor

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

