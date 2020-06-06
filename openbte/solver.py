from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
import deepdish as dd
from mpi4py import MPI
import scipy.sparse as sp
import time
#from matplotlib.pylab import *
import scikits.umfpack as um
from scipy.sparse.linalg import lgmres
import scikits.umfpack as um
import sys


comm = MPI.COMM_WORLD


class Solver(object):

  def __init__(self,**argv):

        #COMMON OPTIONS------------
        self.data = argv
        self.tt = np.float64
        self.state = {}
        self.multiscale = argv.setdefault('multiscale',False)
        self.umfpack = argv.setdefault('umfpack',False)
        #self.multiscale_ballistic = argv.setdefault('multiscale_ballistic',False)
        self.error_multiscale = argv.setdefault('multiscale_error',1e-10)
        self.bundle = argv.setdefault('bundle',False)
        self.verbose = argv.setdefault('verbose',True)
        self.alpha = argv.setdefault('alpha',1.0)
        #self.keep_lu = argv.setdefault('keep_lu',True)
        self.only_fourier = argv.setdefault('only_fourier',False)
        self.max_bte_iter = argv.setdefault('max_bte_iter',20)
        self.max_bte_error = argv.setdefault('max_bte_error',1e-3)
        self.max_fourier_iter = argv.setdefault('max_fourier_iter',20)
        self.max_fourier_error = argv.setdefault('max_fourier_error',1e-5)
        self.init = False
        self.MF = {}
        #----------------------------
         
        if comm.rank == 0:
         if self.verbose: self.print_logo()
         print('                         SYSTEM                 ',flush=True)   
         print(colored(' -----------------------------------------------------------','green'),flush=True)
         self.print_options()

        #-----IMPORT MESH--------------------------------------------------------------
        if comm.rank == 0:
         data = argv['geometry'].data if 'geometry' in argv.keys() else dd.io.load('geometry.h5')
         
         self.n_elems = int(data['meta'][0])
         im = np.concatenate((data['i'],list(np.arange(self.n_elems))))
         jm = np.concatenate((data['j'],list(np.arange(self.n_elems))))
         data['im'] = im
         data['jm'] = jm
         self.assemble_fourier_scalar(data)
        else: data = None
        self.__dict__.update(create_shared_memory_dict(data))
        self.n_elems = int(self.meta[0])
        self.kappa_factor = self.meta[1]
        self.dim = int(self.meta[2])
        self.n_nodes = int(self.meta[3])
        self.n_active_sides = int(self.meta[4])
        if comm.rank == 0 and self.verbose: self.mesh_info() 
        if comm.rank == 0 and self.verbose: self.mpi_info() 
        #------------------------------------------------------------------------------

        #IMPORT MATERIAL---------------------------
        if comm.rank == 0:
         data = argv['material'].data if 'material' in argv.keys() else dd.io.load('material.h5')
         data['VMFP']  *= 1e9
         data['sigma'] *= 1e9
         data['kappa'] = data['kappa'][0:self.dim,0:self.dim] 

         if len(data['tc'].shape) == 1:
          data['B'] = np.einsum('i,ij->ij',data['scale'],data['B'].T + data['B']) 
        else: data = None
        self.__dict__.update(create_shared_memory_dict(data))
        if comm.rank == 0 and self.verbose: self.bulk_info()

        if len(self.tc.shape) == 2:
         self.n_serial = self.tc.shape[0]  
         self.n_parallel = self.tc.shape[1]  
         self.model = 'rta'
         block =  self.n_serial//comm.size
         self.ff = range(block*comm.rank,self.n_serial) if comm.rank == comm.size-1 else range(block*comm.rank,block*(comm.rank+1))
         if comm.rank == 0 and self.verbose: self.mfp_info() 
        else: 
         self.n_parallel = len(self.tc)
         self.model = 'full'

         #tmp = self.B
         #assert np.allclose(tmp, np.tril(tmp))
         #tmp += tmp.T - 0.5*np.diag(np.diag(tmp))
         #self.B = np.einsum('i,ij->ij',self.mat['scale'],tmp)

         if comm.rank == 0 and self.verbose: self.full_info() 
         
        #------------------------------------------

        if comm.rank == 0:
            print(colored(' -----------------------------------------------------------','green'),flush=True)
            print(" ",flush=True)

        if comm.rank == 0:
          #data = self.solve_fourier(self.kappa)
          data = self.solve_fourier_scalar(self.kappa[0,0])

          if data['meta'][0] > self.kappa[0,0]:
              print('WARNING: Fourier thermal conductivity is larger than bulk one.',flush=True)

          variables = {0:{'name':'Temperature Fourier','units':'K',        'data':data['temperature_fourier']},\
                       1:{'name':'Flux Fourier'       ,'units':'W/m/m/K','data':data['flux_fourier']}}
          self.state.update({'variables':variables,\
                           'kappa_fourier':data['meta'][0]})
          self.fourier_info(data)
        else: data = None
        self.__dict__.update(create_shared_memory_dict(data))
       
        #----------------------------------------------------------------
       
        if not self.only_fourier:
         #-------SET Parallel info----------------------------------------------
         block =  self.n_parallel//comm.size
         self.rr = range(block*comm.rank,self.n_parallel) if comm.rank == comm.size-1 else range(block*comm.rank,block*(comm.rank+1))

         #-------SOLVE BTE-------
         data = self.solve_mfp(**argv) if self.model == 'rta' else self.solve_bte(**argv)
         self.state.update({'kappa':data['kappa_vec']})
         #-----------------------

        #Saving-----------------------------------------------------------------------
        if comm.rank == 0:
          variables = self.state['variables']

          if not self.only_fourier:
           variables[2]    = {'name':'Temperature BTE','units':'K'             ,'data':data['temperature']}
           variables[3]    = {'name':'Flux BTE'       ,'units':'W/m/m/K'       ,'data':data['flux']}
           self.state.update({'kappa':data['kappa_vec']})

          if argv.setdefault('save',True):
           if self.bundle:
             self.state['geometry'] = self.mesh
             self.state['material'] = self.mat
           dd.io.save('solver.h5',self.state)
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
          #print(colored('  Multiscale Ballistic:                    ','green')+ str(self.multiscale_ballistic))
          print(colored('  Use umfpack                              ','green')+ str(self.umfpack),flush=True)
          print(colored('  Multiscale Error:                        ','green')+ str(self.error_multiscale),flush=True)
          print(colored('  Only Fourier:                            ','green')+ str(self.only_fourier),flush=True)
          print(colored('  Max Fourier Error:                       ','green')+ '%.1E' % (self.max_fourier_error),flush=True)
          print(colored('  Max Fourier Iter:                        ','green')+ str(self.max_fourier_iter),flush=True)
          print(colored('  Max BTE Error:                           ','green')+ '%.1E' % (self.max_bte_error),flush=True)
          print(colored('  Max BTE Iter:                            ','green')+ str(self.max_bte_iter),flush=True)

  def full_info(self):
          print(colored('  Number of modes:                         ','green')+ str(self.n_parallel),flush=True)

  def mfp_info(self):
          print(colored('  Number of MFP:                           ','green')+ str(self.n_serial),flush=True)
          print(colored('  Number of Solid Angles:                  ','green')+ str(self.n_parallel),flush=True)

  def bulk_info(self):

          #print('                        MATERIAL                 ')   
          #print(colored(' -----------------------------------------------------------','green'))
          print(colored('  Bulk Thermal Conductivity [W/m/K]:       ','green')+ str(round(self.kappa[0,0],2)),flush=True)
          #print(colored(' -----------------------------------------------------------','green'))
          #print(" ")


  def mesh_info(self):

          #print('                        SPACE GRID                 ')   
          #print(colored(' -----------------------------------------------------------','green'))
          print(colored('  Dimension:                               ','green') + str(self.dim),flush=True)
          print(colored('  Size along X [nm]:                       ','green')+ str(self.size[0]),flush=True)
          print(colored('  Size along y [nm]:                       ','green')+ str(self.size[1]),flush=True)
          if self.dim == 3:
           print(colored('  Size along z [nm]:                       ','green')+ str(self.size[2]),flush=True)
          print(colored('  Number of Elements:                      ','green') + str(self.n_elems),flush=True)
          print(colored('  Number of Sides:                         ','green') + str(len(self.active_sides)),flush=True)
          print(colored('  Number of Nodes:                         ','green') + str(len(self.nodes)),flush=True)

          if self.dim == 3:
           filling = np.sum(self.volumes)/self.size[0]/self.size[1]/self.size[2]
          else: 
           filling = np.sum(self.volumes)/self.size[0]/self.size[1]
          print(colored('  Computed porosity:                       ','green') + str(round(1-filling,3)),flush=True)
          
          #print(colored(' -----------------------------------------------------------','green'))
          #print(" ")


  def solve_serial_fourier(self,DeltaT):


        if comm.rank == 0:
         base_old = 0
         fourier = False
         C = np.zeros(self.n_elems)
         for m in range(len(self.mfp_sampled)):
           
            dataf = self.solve_fourier_scalar(self.mfp_average[m],pseudo=DeltaT,m=m,guess = C)
            #C = dataf['C']
            temp  = dataf['temperature_fourier'] 
            grad  = dataf['grad'] 
            base  = np.dot(self.kappa_mask,temp)
            error = abs(base-base_old)/abs(base)
            if error < 1e-3 :
                self.kappaf[m:,:] = -base + np.einsum('m,c,qj,cj->mq',self.mfp_sampled[m:],self.kappa_mask,self.VMFP[:,0:self.dim],grad,optimize=True)
                self.Supd[:] += np.einsum('gq,gqm->m',self.kappaf[:,:],self.suppression[:,:,:,0])*self.kappa_factor*1e-9
                self.tf[m:,:] = temp[:]
                self.tfg[m:,:,:] = grad[:,:]
                break

            else:    
             base_old = base
             self.kappaf[m,:] = -base + self.mfp_sampled[m] * np.einsum('c,qj,cj->q',self.kappa_mask,self.VMFP[:,0:self.dim],grad)
             self.Supd[:] += np.einsum('q,qm->m',self.kappaf[m,:],self.suppression[m,:,:,0])*self.kappa_factor*1e-9
             self.tf[m,:] = temp[:]
             self.tfg[m,:,:] = grad[:,:]
           #---------------------------------

        comm.Barrier()

        return self.kappaf,self.tf,self.tfg,self.Supd


  def solve_modified_fourier(self,DeltaT):

         kappaf,kappafp = np.zeros((2,self.n_serial,self.n_parallel))   
         tf,tfp = np.zeros((2,self.n_serial,self.n_elems))
         tfg,tfgp = np.zeros((2,self.n_serial,self.n_elems,self.dim))
         Supd,Supdp   = np.zeros((2,len(self.kappam)))

         for m in self.ff:
           #dataf = self.solve_fourier(self.mfp_average[m],pseudo=DeltaT,m=m)
           dataf = self.solve_fourier_scalar(self.mfp_average[m],pseudo=DeltaT,m=m,guess = np.zeros(self.n_elems))

           tfp[m] = dataf['temperature_fourier']
           tfgp[m] = dataf['grad']
           for q in range(self.n_parallel): 
            kappafp[m,q] = -np.dot(self.kappa_mask,dataf['temperature_fourier'] - self.mfp_sampled[m]*np.dot(self.VMFP[q],dataf['grad'].T))
            Supdp += kappafp[m,q]*self.suppression[m,q,:,0]*self.kappa_factor*1e-9
         comm.Allreduce([kappafp,MPI.DOUBLE],[kappaf,MPI.DOUBLE],op=MPI.SUM)
         comm.Allreduce([tfp,MPI.DOUBLE],[tf,MPI.DOUBLE],op=MPI.SUM)
         comm.Allreduce([tfgp,MPI.DOUBLE],[tfg,MPI.DOUBLE],op=MPI.SUM)
         comm.Allreduce([Supdp,MPI.DOUBLE],[Supd,MPI.DOUBLE],op=MPI.SUM)


         return kappaf,tf,tfg,Supd

  def solve_modified_fourier_shared(self,DeltaT):

         tf = shared_array(np.zeros((self.n_serial,self.n_elems)) if comm.rank == 0 else None)
         tfg = shared_array(np.zeros((self.n_serial,self.n_elems,self.dim)) if comm.rank == 0 else None)
         kappaf = shared_array(np.zeros((self.n_serial,self.n_elems)) if comm.rank == 0 else None)
 
         for m in self.ff:
           #dataf = self.solve_fourier(self.mfp_average[m],pseudo=DeltaT,m=m)
           dataf = self.solve_fourier_scalar(self.mfp_average[m],pseudo=DeltaT,m=m)
           tf[m,:]    = dataf['temperature_fourier']
           tfg[m,:,:] = dataf['grad']
           for q in range(self.n_parallel): 
            kappaf[m,q] = -np.dot(self.kappa_mask,dataf['temperature_fourier'] - self.mfp_sampled[m]*np.dot(self.VMFP[q],dataf['grad'].T))


         return kappaf,tf,tfg,1


  def get_X_full(self,n,DeltaT,RHSe):  

    if self.init == False:
     self.Master = sp.csc_matrix((np.arange(len(self.im))+1,(self.im,self.jm)),shape=(self.n_elems,self.n_elems),dtype=self.tt)
     self.conversion = np.asarray(self.Master.data-1,np.int) 
     self.init = True
     self.lu = {}
     if self.umfpack:
      self.umfpack = um.UmfpackContext()
      self.umfpack.symbolic(self.Master)

    if not self.umfpack:
      if self.keep_lu:
       if not n in self.lu.keys():
        self.Master.data = np.concatenate((self.Gm[n],self.D[n]+np.ones(self.n_elems)))[self.conversion]
        lu_loc = sp.linalg.splu(self.Master)
        self.lu[n] = lu_loc
       else: lu_loc = self.lu[n]  
      else: 
        self.Master.data = np.concatenate((self.Gm[n],self.D[n]+np.ones(self.n_elems)))[self.conversion]
        lu_loc = sp.linalg.splu(self.Master)


      RHS = np.zeros(self.n_elems)
      for c,i in enumerate(self.eb): RHS[i] += RHSe[m,n,c]

      #RHS = np.zeros(self.n_elems)
      #for c,i in enumerate(self.eb): RHS[i] -= TB[c]*self.Gbm2[n,c]


      X = lu_loc.solve(DeltaT[n] + self.P[n] + RHS)

    else:  
       self.Master.data = np.concatenate((self.Gm[n],self.D[n]+np.ones(self.n_elems)))[self.conversion]
       self.umfpack = um.UmfpackContext()
       self.umfpack.numeric(self.Master)

       RHS = np.zeros(self.n_elems)
       for c,i in enumerate(self.eb): RHS[i] -= TB[c]*self.Gbm2[n,c] 
       X = self.umfpack.solve(um.UMFPACK_A,self.Master,DeltaT[n] + self.P[n] + RHS, autoTranspose = False)

    return X

  #@jit()
  def get_X_jit(self,m,n,DeltaT,RHSe):  
     self.Master = sp.csc_matrix((np.arange(len(self.im))+1,(self.im,self.jm)),shape=(self.n_elems,self.n_elems),dtype=self.tt)
     self.conversion = np.asarray(self.Master.data-1,np.int) 
     self.Master.data = np.concatenate((self.mfp_sampled[m]*self.Gm[n],self.mfp_sampled[m]*self.D[n]+np.ones(self.n_elems)))[self.conversion]
    
     RHS = np.zeros(self.n_elems)
     for c,i in enumerate(self.eb): RHS[i] += RHSe[m,n,c]
     #lu_loc = sp.linalg.splu(self.Master)
     #X = lu_loc.solve(DeltaT + self.mfp_sampled[m]*(self.P[n] + RHS)) 
     
     X = sp.linalg.spsolve(self.Master,DeltaT + self.mfp_sampled[m]*(self.P[n] + RHS))

     return X



  def get_X(self,m,n,DeltaT,RHSe):  


    RHS = np.zeros(self.n_elems)
    for c,i in enumerate(self.eb): RHS[i] += RHSe[m,n,c]
    B = DeltaT + self.mfp_sampled[m]*(self.P[n] + RHS)

    if self.init == False:
     self.Master = sp.csc_matrix((np.arange(len(self.im))+1,(self.im,self.jm)),shape=(self.n_elems,self.n_elems),dtype=self.tt)
     self.conversion = np.asarray(self.Master.data-1,np.int) 
     self.init = True
     self.lu = {}
    if not self.umfpack :
       if not (m,n) in self.lu.keys():
        self.Master.data = np.concatenate((self.mfp_sampled[m]*self.Gm[n],self.mfp_sampled[m]*self.D[n]+np.ones(self.n_elems)))[self.conversion]
        self.lu[(m,n)] = sp.linalg.splu(self.Master)
       return self.lu[(m,n)].solve(B) 
    else:  
      self.Master.data = np.concatenate((self.mfp_sampled[m]*self.Gm[n],self.mfp_sampled[m]*self.D[n]+np.ones(self.n_elems)))[self.conversion]
      return sp.linalg.spsolve(self.Master,B)
      


  def solve_mfp(self,**argv):

     if comm.rank == 0:
           print(flush=True)
           print('      Iter    Thermal Conductivity [W/m/K]      Error ''',flush=True)
           print(colored(' -----------------------------------------------------------','green'),flush=True)   

     #---------------------------------------------i
     if comm.rank == 0:   
      if len(self.db) > 0:
       Gb   = np.einsum('mqj,jn->mqn',self.sigma,self.db,optimize=True)
       Gbp2 = Gb.clip(min=0);
       with np.errstate(divide='ignore', invalid='ignore'):
         tot = 1/Gb.clip(max=0).sum(axis=1); tot[np.isinf(tot)] = 0
       data = {'GG': np.einsum('mqs,ms->mqs',Gbp2,tot)}
       del tot, Gbp2
      else: data = {'GG':np.zeros(1)}
     else: data = None
     self.__dict__.update(create_shared_memory_dict(data))

     #---------------------------------------------
     #------------------------------------------------
     #Main matrix----
     G = np.einsum('qj,jn->qn',self.VMFP[self.rr],self.k,optimize=True)
     Gp = G.clip(min=0); self.Gm = G.clip(max=0)
     self.D = np.zeros((len(self.rr),self.n_elems))
     for n,i in enumerate(self.i):  self.D[:,i] += Gp[:,n]
     DeltaT = self.temperature_fourier.copy()
     #---boundary----------------------
     TB = np.zeros((self.n_serial,len(self.eb)))   
     if len(self.db) > 0: #boundary
      tmp = np.einsum('rj,jn->rn',self.VMFP[self.rr],self.db,optimize=True)  
      Gbp2 = tmp.clip(min=0); self.Gbm2 = tmp.clip(max=0);
      for n,i in enumerate(self.eb):
          self.D[:, i]  += Gbp2[:,n]
          TB[:,n]  = DeltaT[i]  
     #Periodic---
     self.P = np.zeros((len(self.rr),self.n_elems))
     for ss,v in self.pp: self.P[:,self.i[int(ss)]] -= self.Gm[:,int(ss)].copy()*v
     del G,Gp
     #----------------------------------------------------

     lu =  {}
     X = DeltaT.copy()
     kappa_vec = [self.meta[0]]
     kappa_old = kappa_vec[-1]
     error = 1
     kk = 0
     kappa_tot = np.zeros(1)
     MM = np.zeros(2)
     Mp = np.zeros(2)
     kappap = np.zeros((self.n_serial,self.n_parallel))
     kappa = np.zeros((self.n_serial,self.n_parallel))
     termination = True

     self.tf = shared_array(np.zeros((self.n_serial,self.n_elems)) if comm.rank == 0 else None)
     self.tfg = shared_array(np.zeros((self.n_serial,self.n_elems,self.dim)) if comm.rank == 0 else None)
     self.kappaf = shared_array(np.zeros((self.n_serial,self.n_parallel)) if comm.rank == 0 else None)
     self.Supd = shared_array(np.zeros(len(self.kappam)) if comm.rank == 0 else None)

     while kk < self.max_bte_iter and error > self.max_bte_error:
        a = time.time()
        self.x_old = DeltaT.copy()
        if self.multiscale  : 
            a = time.time()
            #(kappaf,tf,tfg,Supd) = self.solve_modified_fourier(DeltaT)
            (kappaf,tf,tfg,Supd) = self.solve_serial_fourier(DeltaT)
            #(kappaf,tf,tfg,Supd) = self.solve_modified_fourier_shared(DeltaT)
            #print(time.time()-a,'FF',comm.rank)

        #Multiscale scheme-----------------------------
        diffusive = 0
        bal = 0
        DeltaTp = np.zeros_like(DeltaT)
        Supbp = np.zeros_like(self.kappam)
        TBp = np.zeros_like(TB)
        Sup,Supp   = np.zeros((2,len(self.kappam)))
        Supb,Supbp = np.zeros((2,len(self.kappam)))
        J,Jp = np.zeros((2,self.n_elems,self.dim))
        #kappa_bal,kappa_balp = np.zeros((2,self.n_parallel))

        #EXPERIMENTAL--
        RHS = -np.einsum('mc,nc->mnc',TB,self.Gbm2)

        for n,q in enumerate(self.rr):
           #COMPUTE BALLISTIC----------------------- 
           if self.multiscale:
              X_bal = self.get_X(-1,n,DeltaT,RHS)
              kappa_bal = -np.dot(self.kappa_mask,X_bal)
              Supbp += kappa_bal*self.suppression[:,q,:,0].sum(axis=0)*self.kappa_factor*1e-9
              idx  = np.argwhere(np.diff(np.sign(kappaf[:,q] - kappa_bal*np.ones(self.n_serial)))).flatten()
              if len(idx) == 0: idx = [self.n_serial-1]
           else: idx = [self.n_serial-1]
         
           fourier = False
           for m in range(self.n_serial)[idx[0]::-1]:

             #----------------------------------   
             if fourier:
               kappap[m,q] = kappaf[m,q]
               X = tf[m] - self.mfp_sampled[m]*np.dot(self.VMFP[q,:self.dim],tfg[m].T)
               diffusive +=1
             else:
               X = self.get_X(m,n,DeltaT,RHS)
               kappap[m,q] = -np.dot(self.kappa_mask,X)
               if self.multiscale:
                if abs(kappap[m,q] - kappaf[m,q])/abs(kappap[m,q]) < self.error_multiscale :
                  kappap[m,q] = kappaf[m,q]
                  diffusive +=1
                  fourier=True

             #-------------------------------------     
             DeltaTp += X*self.tc[m,q]
             for c,i in enumerate(self.eb): TBp[m,c] -= X[i]*self.GG[m,q,c]
             Jp += np.einsum('c,j->cj',X,self.sigma[m,q,0:self.dim])*1e-18
             Supp += kappap[m,q]*self.suppression[m,q,:,0]*self.kappa_factor*1e-9
 
           ballistic = False
           for m in range(self.n_serial)[idx[0]+1:]:
              if ballistic:
               kappap[m,q] = kappa_bal
               X = X_bal
               bal +=1
              else: 
               X = self.get_X(m,n,DeltaT,RHS)
               kappap[m,q] = -np.dot(self.kappa_mask,X)
               if abs(kappap[m,q] - kappa_bal)/abs(kappap[m,q]) < self.error_multiscale :
                   kappap[m,q] = kappa_bal
                   bal +=1
                   ballistic=True

              DeltaTp += X*self.tc[m,q]
              for c,i in enumerate(self.eb): TBp[m,c] -= X[i]*self.GG[m,q,c]
              Jp += np.einsum('c,j->cj',X,self.sigma[m,q,0:self.dim])*1e-18
              Supp += kappap[m,q]*self.suppression[m,q,:,0]*self.kappa_factor*1e-9
           
        Mp[0] = diffusive
        Mp[1] = bal

        comm.Barrier()
        comm.Allreduce([DeltaTp,MPI.DOUBLE],[DeltaT,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Supp,MPI.DOUBLE],[Sup,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Supbp,MPI.DOUBLE],[Supb,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Jp,MPI.DOUBLE],[J,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Mp,MPI.DOUBLE],[MM,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([TBp,MPI.DOUBLE],[TB,MPI.DOUBLE],op=MPI.SUM)
        kappa_totp = np.array([np.einsum('mq,mq->',self.sigma[:,self.rr,0],kappa[:,self.rr])])*self.kappa_factor*1e-18
        comm.Allreduce([kappa_totp,MPI.DOUBLE],[kappa_tot,MPI.DOUBLE],op=MPI.SUM)
        kk +=1
        #print(time.time()-a)
        error = abs(kappa_old-kappa_tot[0])/abs(kappa_tot[0])
        kappa_old = kappa_tot[0]
        kappa_vec.append(kappa_tot[0])
        if self.verbose and comm.rank == 0:   
         print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa_vec[-1],error),flush=True)

        #if comm.rank == 0:
        #  diff = int(MM[0])/self.n_serial/self.n_parallel 
        #  bal = int(MM[1])/self.n_serial/self.n_parallel 
        #  print(colored(' BTE:              ','green') + str(round((1-diff-bal)*100,2)) + ' %' )
        #  print(colored(' FOURIER:          ','green') + str(round(diff*100,2)) + ' %' )
        #  print(colored(' BALLISTIC:          ','green') + str(round(bal*100,2)) + ' %' )
        #  print(colored(' Full termination: ','green') + str(termination) )
        #  print(colored(' -----------------------------------------------------------','green'))
          #plot(self.mfp_bulk,Sup,color='b',marker='o')
          #if self.multiscale:
          # plot(self.mfp_bulk,Supd,color='r',marker='o')
          #if self.multiscale_ballistic:
          # plot(self.mfp_bulk,Supb,color='g',marker='o')
          #ylim([0,1])
          #xscale('log')
          #show()
        #-----------------------------------------------

     if self.verbose and comm.rank == 0:
      print(colored(' -----------------------------------------------------------','green'),flush=True)

     if self.multiscale and comm.rank == 0:
        print(flush=True)
        print('                  Multiscale Diagnostics        ''',flush=True)
        print(colored(' -----------------------------------------------------------','green'),flush=True)

        diff = int(MM[0])/self.n_serial/self.n_parallel 
        bal = int(MM[1])/self.n_serial/self.n_parallel 
        print(colored(' BTE:              ','green') + str(round((1-diff-bal)*100,2)) + ' %',flush=True)
        print(colored(' FOURIER:          ','green') + str(round(diff*100,2)) + ' %',flush=True)
        print(colored(' BALLISTIC:        ','green') + str(round(bal*100,2)) + ' %',flush=True)
        print(colored(' Full termination: ','green') + str(termination),flush=True)
        print(colored(' -----------------------------------------------------------','green'),flush=True)

        #plot(self.mfp_bulk,Sup,color='b',marker='o')
        #if self.multiscale:
        # plot(self.mfp_bulk,Supd,color='r',marker='o')
        #if self.multiscale_ballistic:
        # plot(self.mfp_bulk,Supb,color='g',marker='o')
        #ylim([0,1])
        #xscale('log')
        #show()
        #-----------------------------------------------



     #return {'kappa_vec':kappa_vec,'temperature':DeltaT,'flux':J,'suppression':Sup,'suppression_diffusive':supd,'suppression_ballistic':Supb}
     output =  {'kappa_vec':kappa_vec,'temperature':DeltaT,'flux':J}
     
     if self.multiscale:           output.update({'suppression_diffusive':Supd,'suppression_ballistic':Supb})

     return output


  def solve_bte(self,**argv):

     if comm.rank == 0:
           print(flush=True)
           print('      Iter    Thermal Conductivity [W/m/K]      Error ''',flush=True)
           print(colored(' -----------------------------------------------------------','green'),flush=True)   


     if comm.rank == 0:   
      if len(self.db) > 0:
       Gb   = np.einsum('qj,jn->qn',self.sigma,self.db,optimize=True)
       Gbp2 = Gb.clip(min=0);
       with np.errstate(divide='ignore', invalid='ignore'):
         tot = 1/Gb.clip(max=0).sum(axis=0); tot[np.isinf(tot)] = 0
       data = {'GG': np.einsum('qs,s->qs',Gbp2,tot)}
       del tot, Gbp2
      else: data = {'GG':np.zeros(1)}
     else: data = None
     self.__dict__.update(create_shared_memory_dict(data))

     #Main matrix----
     G = np.einsum('qj,jn->qn',self.VMFP[self.rr],self.k,optimize=True)
     Gp = G.clip(min=0); self.Gm = G.clip(max=0)
     self.D = np.zeros((len(self.rr),self.n_elems))
     for n,i in enumerate(self.i):  self.D[:,i] += Gp[:,n]
     #---boundary----------------------
     #DeltaT = self.temperature_fourier.copy()
     TB = np.zeros(len(self.eb))
     if len(self.db) > 0: #boundary
      tmp = np.einsum('rj,jn->rn',self.VMFP[self.rr],self.db,optimize=True)  
      Gbp2 = tmp.clip(min=0); self.Gbm2 = tmp.clip(max=0);
      for n,i in enumerate(self.eb):
          self.D[:, i]  += Gbp2[:,n]
          #TB[n]  = DeltaT[i]  
          TB[n]  = self.temperature_fourier[i] 
     #Periodic---
     self.P = np.zeros((len(self.rr),self.n_elems))
     for ss,v in self.pp: self.P[:,self.i[int(ss)]] -= self.Gm[:,int(ss)].copy()*v
     del G,Gp
     #----------------------------------------------------
     

     kappa_vec = [self.meta[0]]
     kappa_old = kappa_vec[-1]
     error = 1
     kk = 0
     kappa_tot = np.zeros(1)
     MM = np.zeros(2)
     Mp = np.zeros(2)
     kappap,kappa = np.zeros((2,self.n_parallel))
     termination = True
     alpha = 1

     X = np.tile(self.temperature_fourier,(self.n_parallel,1))
     X_old = X.copy()
     #Xs = shared_array(np.tile(self.temperature_fourier,(self.n_parallel,1)) if comm.rank == 0 else None)
     #Xs_old = shared_array(np.tile(self.temperature_fourier,(self.n_parallel,1)) if comm.rank == 0 else None)
     #Xs_old[:,:] = Xs[:,:].copy()

     while kk < self.max_bte_iter and error > self.max_bte_error:

      DeltaT = np.matmul(self.B[self.rr],alpha*X+(1-alpha)*X_old)
      X_old = X.copy()
      #DeltaTs = np.matmul(self.B[self.rr],alpha*Xs+(1-alpha)*Xs_old)


      RHS = -np.einsum('mc,nc->mnc',TB,self.Gbm2)
      TBp = np.zeros_like(TB)
      Xp = np.zeros_like(X)
      J,Jp = np.zeros((2,self.n_elems,3))
      kappa,kappap = np.zeros((2,self.n_parallel))
      for n,q in enumerate(self.rr):
         Xp[q] = self.get_X_full(n,DeltaT,RHS)

         for c,i in enumerate(self.eb): TBp[c] -= Xp[q,i]*self.GG[q,c]
         kappap[q] -= np.dot(self.kappa_mask,Xp[q])

         #Xs[q,:] = self.get_X_full(n,TB,DeltaTs)
         #for c,i in enumerate(self.eb): TBp[c] -= Xs[q,i]*self.GG[q,c]
         #kappap[q] -= np.dot(self.kappa_mask,Xs[q])
       
      comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
      comm.Allreduce([Xp,MPI.DOUBLE],[X,MPI.DOUBLE],op=MPI.SUM)
      comm.Allreduce([TBp,MPI.DOUBLE],[TB,MPI.DOUBLE],op=MPI.SUM)
      kappa_totp = np.array([np.einsum('q,q->',self.sigma[self.rr,0],kappa[self.rr])])*self.kappa_factor*1e-18
      comm.Allreduce([kappa_totp,MPI.DOUBLE],[kappa_tot,MPI.DOUBLE],op=MPI.SUM)
      comm.Barrier()

     
      error = abs(kappa_old-kappa_tot[0])/abs(kappa_tot[0])
      kappa_old = kappa_tot[0]
      kappa_vec.append(kappa_tot[0])
      if self.verbose and comm.rank == 0:   
        print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa_vec[-1],error),flush=True)
      kk+=1

     if self.verbose and comm.rank == 0:
      print(colored(' -----------------------------------------------------------','green'),flush=True)

     
     #T = np.einsum('qc,q->c',Xs,self.tc)
     #J = np.einsum('qj,qc->cj',self.sigma,Xs)*1e-18
     T = np.einsum('qc,q->c',X,self.tc)
     J = np.einsum('qj,qc->cj',self.sigma,X)*1e-18
     return {'kappa_vec':kappa_vec,'temperature':T,'flux':J}



  def get_decomposed_directions(self,ll,rot):

    normal = self.face_normals[ll,0:self.dim]
    dist   = self.dists[ll,0:self.dim]
    v_orth = np.dot(normal,np.dot(rot,normal))/np.dot(normal,dist)
    v_non_orth = np.dot(rot,normal) - dist*v_orth

    return v_orth,v_non_orth[:self.dim]


  def get_kappa(self,i,j,ll,kappa):

   if i ==j:
    return np.array(kappa[i])
   
   normal = self.face_normals[ll,0:self.dim]


   kappa_i = np.array(kappa[i])
   kappa_j = np.array(kappa[j])

   ki = np.dot(normal,np.dot(kappa_i,normal))
   kj = np.dot(normal,np.dot(kappa_j,normal))
   w  = self.interp_weigths[ll]

   kappa_loc = kj*kappa_i/(ki*(1-w) + kj*w)
 
   return kappa_loc

  

  def assemble_fourier_scalar(self,mesh):
    
    iff = []
    jff = []
    dff = []

    B = np.zeros(self.n_elems)
    for ll in mesh['active_sides']:

      area = mesh['areas'][ll] 
      (i,j) = mesh['side_elem_map_vec'][ll]
      vi = mesh['volumes'][i]
      vj = mesh['volumes'][j]
      if not i == j:

       normal = mesh['face_normals'][ll]
       dist   = mesh['dists'][ll]
       v_orth = 1/np.dot(normal,dist)

       iff.append(i)
       jff.append(i)
       dff.append(v_orth/vi*area)
       iff.append(i)
       jff.append(j)
       dff.append(-v_orth/vi*area)
       iff.append(j)
       jff.append(j)
       dff.append(v_orth/vj*area)
       iff.append(j)
       jff.append(i)
       dff.append(-v_orth/vj*area)
       if ll in mesh['periodic_sides']:    
        kk = list(mesh['periodic_sides']).index(ll)   
        B[i] += mesh['periodic_side_values'][kk]*v_orth/vi*area
        B[j] -= mesh['periodic_side_values'][kk]*v_orth/vj*area

    mesh['RHS_FOURIER'] =B   
    mesh['iff'] = np.array(iff)
    mesh['jff'] = np.array(jff)
    mesh['dff'] = np.array(dff)
    

  def solve_fourier_scalar(self,kappa,**argv):

    m = argv.setdefault('m',-1)

    if m in self.MF.keys():
       SU = self.MF[m]['SU']
       scale = self.MF[m]['scale']
       B = self.MF[m]['B'] + argv['pseudo']
    else: 
     F = sp.csc_matrix((self.dff,(self.iff,self.jff)),shape = (self.n_elems,self.n_elems))
     if 'pseudo' in argv.keys():
       a = time.time() 
       F = kappa*F + sp.eye(self.n_elems)
       scale = 1/F.max(axis=0).toarray()[0]
       F.data = F.data * scale[F.indices]
       SU = splu(F)
       B = kappa*self.RHS_FOURIER.copy()
       self.MF[m] = {'SU':SU,'scale':scale,'B':B.copy()}
       B = B + argv['pseudo']
     else:  
      F *= kappa   
      B = self.RHS_FOURIER.copy()*kappa
      scale = 1/F.max(axis=0).toarray()[0]
      n = np.random.randint(self.n_elems)
      scale[n] = 0
      F.data = F.data * scale[F.indices]
      F[n,n] = 1
      B[n] = 0
      SU = splu(F)

    #--------------
    C = np.zeros(self.n_elems)
    #C = argv['guess']
    n_iter = 0
    kappa_old = 0
    error = 1  
    grad = np.zeros((self.n_elems,self.dim))
    while error > self.max_fourier_error and \
                  n_iter < self.max_fourier_iter :

        RHS = B + C
        for n in range(self.n_elems):
          RHS[n] = RHS[n]*scale[n]  

        temp = SU.solve(RHS)
        temp = temp - (max(temp)+min(temp))/2.0
        kappa_eff = self.compute_diffusive_thermal_conductivity(temp,grad,kappa)

        error = abs((kappa_eff - kappa_old)/kappa_eff)
        kappa_old = kappa_eff
        n_iter +=1
        #grad = self.compute_grad(temp,grad)
        #C = self.compute_non_orth_contribution(grad,kappa)
        C,grad = self.compute_secondary_flux(temp,kappa)
        #print(np.allclose(C2,C))

    flux = -kappa*grad

    meta = [kappa_eff,error,n_iter] 
    return {'flux_fourier':flux,'temperature_fourier':temp,'meta':np.array(meta),'grad':grad,'C':C}

  def compute_secondary_flux(self,temp,kappa):

   if not np.isscalar(kappa):
     kappa = kappa[0,0]

   diff_temp = np.zeros((self.n_elems, self.n_side_per_elem[0]))

   a = time.time()
   for ll in self.active_sides :
    elems = self.side_elem_map_vec[ll]
    kc1 = elems[0]
    c1 = self.centroids[kc1]

    ind1 = list(self.elem_side_map_vec[kc1]).index(ll)
    if not ll in self.boundary_sides: 
     kc2 = elems[1]
     ind2 = list(self.elem_side_map_vec[kc2]).index(ll)
     temp_1 = temp[kc1]
     temp_2 = temp[kc2]

     if ll in self.periodic_sides:
      temp_2 += self.periodic_side_values[list(self.periodic_sides).index(ll)]

     diff_t = temp_2 - temp_1
     diff_temp[kc1][ind1]  = diff_t
     diff_temp[kc2][ind2]  = -diff_t

   gradT = np.einsum('kjs,ks->kj',self.weigths,diff_temp)

   #-----------------------------------------------------
   F_ave = np.zeros((len(self.sides),self.dim))
   for ll in self.active_sides:
      (i,j) = self.side_elem_map_vec[ll]
      if not i==j:
       w = self.interp_weigths[ll]
       F_ave[ll] = w*gradT[i] + (1.0-w)*gradT[j]
   F_ave *= kappa 

   #a = time.time()
   C = np.zeros(self.n_elems)
   for ll in self.active_sides:
    (i,j) = self.side_elem_map_vec[ll]
    if not i==j:
      area = self.areas[ll]   
      (dummy,v_non_orth) = self.get_decomposed_directions(ll,np.eye(self.dim))#,rot=self.mat['kappa'])
      tmp = np.dot(F_ave[ll],v_non_orth)*area
      C[i] += tmp/self.volumes[i]
      C[j] -= tmp/self.volumes[j]

   #print(time.time()-a)   

   return C,gradT


  def solve_fourier(self,kappa,**argv):

   if np.isscalar(kappa):
       kappa = np.diag(np.diag(kappa*np.eye(self.dim)))

   if kappa.ndim == 2:
      kappa = np.repeat(np.array([np.diag(np.diag(kappa))]),self.n_elems,axis=0)

   m = argv.setdefault('m',-1)
   if m in self.MF.keys():
       SU = self.MF[m]['SU']
       scale = self.MF[m]['scale']
       B = self.MF[m]['B'] + argv['pseudo']
   else:   

    F = sp.dok_matrix((self.n_elems,self.n_elems))
    B = np.zeros(self.n_elems)
    for ll in self.active_sides:

      area = self.areas[ll] 
      (i,j) = self.side_elem_map_vec[ll]
      vi = self.volumes[i]
      vj = self.volumes[j]
      kappa_loc = self.get_kappa(i,j,ll,kappa)
      if not i == j:
       (v_orth,dummy) = self.get_decomposed_directions(ll,rot=kappa_loc)
       F[i,i] += v_orth/vi*area
       F[i,j] -= v_orth/vi*area
       F[j,j] += v_orth/vj*area
       F[j,i] -= v_orth/vj*area
       if ll in self.periodic_sides:    
        kk = list(self.periodic_sides).index(ll)   
        B[i] += self.periodic_side_values[kk]*v_orth/vi*area
        B[j] -= self.periodic_side_values[kk]*v_orth/vj*area
   
    
    #rescaleand fix one point to 0
    F = F.tocsc()
    if 'pseudo' in argv.keys():
       F = F + sp.eye(self.n_elems)
       scale = 1/F.max(axis=0).toarray()[0]
       F.data = F.data * scale[F.indices]
       SU = splu(F)
       self.MF[m] = {'SU':SU,'scale':scale,'B':B}
       B = B + argv['pseudo']
    else:  
      scale = 1/F.max(axis=0).toarray()[0]
      n = np.random.randint(self.n_elems)
      scale[n] = 0
      F.data = F.data * scale[F.indices]
      F[n,n] = 1
      B[n] = 0
      SU = splu(F)
    #-----------------------

   C = np.zeros(self.n_elems)
   n_iter = 0
   kappa_old = 0
   error = 1  
   grad = np.zeros((self.n_elems,self.dim))
   while error > self.max_fourier_error and \
                  n_iter < self.max_fourier_iter :
        RHS = B + C
        for n in range(self.n_elems):
          RHS[n] = RHS[n]*scale[n]  

        temp = SU.solve(RHS)
        temp = temp - (max(temp)+min(temp))/2.0
        kappa_eff = self.compute_diffusive_thermal_conductivity(temp,grad,kappa)
        error = abs((kappa_eff - kappa_old)/kappa_eff)
        kappa_old = kappa_eff
        n_iter +=1
        #grad = self.compute_grad(temp,grad)
        #C = self.compute_non_orth_contribution(grad,kappa)

        C,grad = self.compute_secondary_flux(temp,kappa)


   flux = -np.einsum('cij,cj->ci',kappa,grad)
 
   meta = [kappa_eff,error,n_iter] 
   return {'flux_fourier':flux,'temperature_fourier':temp,'meta':np.array(meta),'grad':grad}

   #return {'flux':flux,'temperature':temp,'kappa':kappa_eff,'grad':grad,'error':error,'n_iter':n_iter}

  def compute_grad(self,temp,gradT):

   diff_temp = self.n_elems*[None]
   for i in range(len(diff_temp)):
      diff_temp[i] = self.n_side_per_elem[i]*[0] 

   for ll in self.active_sides :
    elems = self.side_elem_map_vec[ll]
    kc1 = elems[0]
    c1 = self.centroids[kc1]

    ind1 = list(self.elem_side_map_vec[kc1]).index(ll)
    if not ll in self.boundary_sides: 
     kc2 = elems[1]
     ind2 = list(self.elem_side_map_vec[kc2]).index(ll)
     temp_1 = temp[kc1]
     temp_2 = temp[kc2]

     if ll in self.periodic_sides:
      temp_2 += self.periodic_side_values[list(self.periodic_sides).index(ll)]

     diff_t = temp_2 - temp_1
     diff_temp[kc1][ind1]  = diff_t
     diff_temp[kc2][ind2]  = -diff_t

   gradT = np.zeros((self.n_elems,self.dim))
   for k in range(self.n_elems) :
    tmp = np.dot(self.weigths[k],diff_temp[k])
    gradT[k,0] = tmp[0] #THESE HAS TO BE POSITIVE
    gradT[k,1] = tmp[1]
    if self.dim == 3:
     gradT[k,2] = tmp[2]

   return gradT  


  def compute_non_orth_contribution(self,gradT,kappa) :

    C = np.zeros(self.n_elems)

    for ll in self.active_sides:

     (i,j) = self.side_elem_map_vec[ll]

     if not i==j:

      if np.isscalar(kappa):
        kappa_1 = np.diag(np.diag(kappa*np.eye(self.dim)))
        kappa_2 = np.diag(np.diag(kappa*np.eye(self.dim)))
      else:  
        kappa_1 = kappa[i]  
        kappa_2 = kappa[j]  

      area = self.areas[ll]   
      w = self.interp_weigths[ll]

      #print(np.shape(gradT))
      #print(np.shape(kappa_1))
      #print(np.shape(kappa_2))
      F_ave = w*np.dot(gradT[i],kappa_1) + (1.0-w)*np.dot(gradT[j],kappa_2)
      #grad_ave = w*gradT[i] + (1.0-w)*gradT[j]

      (dummy,v_non_orth) = self.get_decomposed_directions(ll,np.eye(self.dim))#,rot=self.mat['kappa'])
      

      C[i] += np.dot(F_ave,v_non_orth)/self.volumes[i]*area
      C[j] -= np.dot(F_ave,v_non_orth)/self.volumes[j]*area

    return C


  def compute_diffusive_thermal_conductivity(self,temp,gradT,kappa):

   kappa_eff = 0
   for l in self.flux_sides:

    (i,j) = self.side_elem_map_vec[l]

    
    if np.isscalar(kappa):
      kappa = np.diag(np.diag(kappa*np.eye(self.dim)))
 
    if kappa.ndim == 3:
      kappa = self.get_kappa(i,j,l,kappa)
       
    (v_orth,v_non_orth) = self.get_decomposed_directions(l,rot=kappa)

    deltaT = temp[i] - (temp[j] + 1)
    kappa_eff -= v_orth *  deltaT * self.areas[l]
    w  = self.interp_weigths[l]
    grad_ave = w*gradT[i] + (1.0-w)*gradT[j]

    kappa_eff += np.dot(grad_ave,v_non_orth)/2 * self.areas[l]


   return kappa_eff*self.kappa_factor

  def print_logo(self):


    #v = pkg_resources.require("OpenBTE")[0].version   
    print(' ',flush=True)
    print(colored(r'''        ___                   ____ _____ _____ ''','green'),flush=True)
    print(colored(r'''       / _ \ _ __   ___ _ __ | __ )_   _| ____|''','green'),flush=True)
    print(colored(r'''      | | | | '_ \ / _ \ '_ \|  _ \ | | |  _|  ''','green'),flush=True)
    print(colored(r'''      | |_| | |_) |  __/ | | | |_) || | | |___ ''','green'),flush=True)
    print(colored(r'''       \___/| .__/ \___|_| |_|____/ |_| |_____|''','green'),flush=True)
    print(colored(r'''            |_|                                ''','green'),flush=True)
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

