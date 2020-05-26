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

comm = MPI.COMM_WORLD


class Solver(object):

  def __init__(self,**argv):

        #COMMON OPTIONS------------
        self.data = argv
        self.tt = np.float64
        self.state = {}
        self.multiscale = argv.setdefault('multiscale',False)
        self.umfpack = argv.setdefault('umfpack',False)
        self.multiscale_ballistic = argv.setdefault('multiscale_ballistic',False)
        self.error_multiscale = argv.setdefault('multiscale_error',1e-10)
        self.verbose = argv.setdefault('verbose',True)
        self.alpha = argv.setdefault('alpha',1.0)
        self.keep_lu = argv.setdefault('keep_lu',True)
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
         print('                         SYSTEM                 ')   
         print(colored(' -----------------------------------------------------------','green'))
         self.print_options()

        #-----IMPORT MESH--------------------------------------------------------------
        if comm.rank == 0:
         data = argv['geometry'].data if 'geometry' in argv.keys() else dd.io.load('geometry.h5')
         n_elems = int(data['meta'][0])
         im = np.concatenate((data['i'],list(np.arange(n_elems))))
         jm = np.concatenate((data['j'],list(np.arange(n_elems))))
         data['im'] = im
         data['jm'] = jm
        else: data = None
        self.__dict__.update(create_shared_memory_dict(data))
        self.n_elems = int(self.meta[0])
        self.kappa_factor = self.meta[1]
        self.dim = int(self.meta[2])
        self.n_nodes = int(self.meta[3])
        self.n_active_sides = int(self.meta[4])
        if comm.rank == 0 and self.verbose: self.mesh_info() 
        #------------------------------------------------------------------------------

        #IMPORT MATERIAL---------------------------
        if comm.rank == 0:
         data = argv['material'].data if 'material' in argv.keys() else dd.io.load('material.h5')
         data['VMFP']  *= 1e9
         data['sigma'] *= 1e9
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
            print(colored(' -----------------------------------------------------------','green'))
            print(" ")

        if comm.rank == 0:
          data = self.solve_fourier(self.kappa)

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
         #-----------------------

        #Saving-----------------------------------------------------------------------
        if comm.rank == 0:
          variables = self.state['variables']

          if not self.only_fourier:
           variables[2]    = {'name':'Temperature BTE','units':'K'             ,'data':data['temperature']}
           variables[3]    = {'name':'Flux BTE'       ,'units':'W/m/m/K'       ,'data':data['flux']}

          self.state.update({'variables':variables})#,\
                           #'kappa':data['kappa_vec'],\
                           #'suppression':data['suppression'],\
                           #'suppression_diffusive':data['suppression_diffusive']})

          

          if argv.setdefault('save',True):
           dd.io.save('solver.h5',self.state)
          if self.verbose:
           print(' ')   
           print(colored('                 OpenBTE ended successfully','green'))
           print(' ')  


  def fourier_info(self,data):


          print('                        FOURIER                 ')   
          print(colored(' -----------------------------------------------------------','green'))
          print(colored('  Iterations:                              ','green') + str(int(data['meta'][2])))
          print(colored('  Relative error:                          ','green') + '%.1E' % (data['meta'][1]))
          print(colored('  Fourier Thermal Conductivity [W/m/K]:    ','green') + str(round(data['meta'][0],3)))
          print(colored(' -----------------------------------------------------------','green'))
          print(" ")



  def print_options(self):
          print(colored('  Multiscale Diffusive:                    ','green')+ str(self.multiscale))
          print(colored('  Multiscale Ballistic:                    ','green')+ str(self.multiscale_ballistic))
          print(colored('  Keep LU                                  ','green')+ str(self.keep_lu))
          print(colored('  Multiscale Error:                        ','green')+ str(self.error_multiscale))
          print(colored('  Only Fourier:                            ','green')+ str(self.only_fourier))
          print(colored('  Max Fourier Error:                       ','green')+ '%.1E' % (self.max_fourier_error))
          print(colored('  Max Fourier Iter:                        ','green')+ str(self.max_fourier_iter))
          print(colored('  Max BTE Error:                           ','green')+ '%.1E' % (self.max_bte_error))
          print(colored('  Max BTE Iter:                            ','green')+ str(self.max_bte_iter))

  def full_info(self):
          print(colored('  Number of modes:                         ','green')+ str(self.n_parallel))

  def mfp_info(self):
          print(colored('  Number of MFP:                           ','green')+ str(self.n_serial))
          print(colored('  Number of Solid Angles:                  ','green')+ str(self.n_parallel))

  def bulk_info(self):

          #print('                        MATERIAL                 ')   
          #print(colored(' -----------------------------------------------------------','green'))
          print(colored('  Bulk Thermal Conductivity [W/m/K]:       ','green')+ str(round(self.kappa[0,0],4)))
          #print(colored(' -----------------------------------------------------------','green'))
          #print(" ")


  def mesh_info(self):

          #print('                        SPACE GRID                 ')   
          #print(colored(' -----------------------------------------------------------','green'))
          print(colored('  Dimension:                               ','green') + str(self.dim))
          print(colored('  Size along X [nm]:                       ','green')+ str(self.size[0]))
          print(colored('  Size along y [nm]:                       ','green')+ str(self.size[1]))
          if self.dim == 3:
           print(colored('  Size along z [nm]:                       ','green')+ str(self.size[0]))
          print(colored('  Number of Elements:                      ','green') + str(self.n_elems))
          print(colored('  Number of Sides:                         ','green') + str(len(self.active_sides)))
          print(colored('  Number of Nodes:                         ','green') + str(len(self.nodes)))
          #print(colored(' -----------------------------------------------------------','green'))
          #print(" ")



  def solve_modified_fourier(self,DeltaT):

         kappaf,kappafp = np.zeros((2,self.n_serial,self.n_parallel))   
         tf,tfp = np.zeros((2,self.n_serial,self.n_elems))
         tfg,tfgp = np.zeros((2,self.n_serial,self.n_elems,3))
         Supd,Supdp   = np.zeros((2,len(self.kappam)))

         for m in self.ff:
           dataf = self.solve_fourier(self.mfp_average[m],pseudo=DeltaT,m=m)
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

         #Supd,Supdp   = np.zeros((2,len(self.kappam)))
         tf = shared_array(np.zeros((self.n_serial,self.n_elems)) if comm.rank == 0 else None)
         tfg = shared_array(np.zeros((self.n_serial,self.n_elems,3)) if comm.rank == 0 else None)
         kappaf = shared_array(np.zeros((self.n_serial,self.n_elems)) if comm.rank == 0 else None)
         #Supd = shared_array(np.zeros(len(self.kappam)) if comm.rank == 0 else None)

 
         a = time.time()
         for m in self.ff:
           dataf = self.solve_fourier(self.mfp_average[m],pseudo=DeltaT,m=m)
           tf[m,:]    = dataf['temperature_fourier']
           tfg[m,:,:] = dataf['grad']
           for q in range(self.n_parallel): 
            kappaf[m,q] = -np.dot(self.kappa_mask,dataf['temperature_fourier'] - self.mfp_sampled[m]*np.dot(self.VMFP[q],dataf['grad'].T))
            #Supdp[:] += kappafp[m,q]*self.suppression[m,q,:,0]*self.kappa_factor*1e-9
         print(comm.rank,time.time()-a)

         #a = time.time()
         #comm.Allreduce([Supdp,MPI.DOUBLE],[Supd,MPI.DOUBLE],op=MPI.SUM)

         #print(time.time()-a)

         return kappaf,tf,tfg,1


  def get_X_full(self,n,TB,DeltaT):  

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
      for c,i in enumerate(self.eb): RHS[i] -= TB[c]*self.Gbm2[n,c]
      #X = lu_loc.solve(DeltaT[self.rr[n]] + self.P[n] + RHS)
      X = lu_loc.solve(DeltaT[n] + self.P[n] + RHS)

    else:  
       self.Master.data = np.concatenate((self.Gm[n],self.D[n]+np.ones(self.n_elems)))[self.conversion]
       self.umfpack = um.UmfpackContext()
       self.umfpack.numeric(self.Master)

       RHS = np.zeros(self.n_elems)
       for c,i in enumerate(self.eb): RHS[i] -= TB[c]*self.Gbm2[n,c] 
       X = self.umfpack.solve(um.UMFPACK_A,self.Master,DeltaT[n] + self.P[n] + RHS, autoTranspose = False)

    return X

  def get_X(self,m,n,TB,DeltaT):  

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
       if not (m,n) in self.lu.keys():
        self.Master.data = np.concatenate((self.mfp_sampled[m]*self.Gm[n],self.mfp_sampled[m]*self.D[n]+np.ones(self.n_elems)))[self.conversion]
        lu_loc = sp.linalg.splu(self.Master)
        self.lu[(m,n)] = lu_loc
       else: lu_loc = self.lu[(m,n)]  
      else: 
        self.Master.data = np.concatenate((self.mfp_sampled[m]*self.Gm[n],self.mfp_sampled[m]*self.D[n]+np.ones(self.n_elems)))[self.conversion]
        lu_loc = sp.linalg.splu(self.Master)

      RHS = np.zeros(self.n_elems)
      for c,i in enumerate(self.eb): RHS[i] -= TB[m,c]*self.Gbm2[n,c]
      X = lu_loc.solve(DeltaT + self.mfp_sampled[m]*(self.P[n] + RHS)) 

    else:  
       self.Master.data = np.concatenate((self.mfp_sampled[m]*self.Gm[n],self.mfp_sampled[m]*self.D[n]+np.ones(self.n_elems)))[self.conversion]
       self.umfpack = um.UmfpackContext()
       self.umfpack.numeric(self.Master)

       RHS = np.zeros(self.n_elems)
       for c,i in enumerate(self.eb): RHS[i] -= TB[m,c]*self.Gbm2[n,c] 
       X = self.umfpack.solve(um.UMFPACK_A,self.Master,DeltaT  + self.mfp_sampled[m]*(self.P[n] + RHS), autoTranspose = False)

    return X

  def solve_mfp(self,**argv):


     if comm.rank == 0:
           print()
           print('      Iter    Thermal Conductivity [W/m/K]      Error ''')
           print(colored(' -----------------------------------------------------------','green'))   

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

     while kk < self.max_bte_iter and error > self.max_bte_error:
        
        if self.multiscale  : 
            (kappaf,tf,tfg,Supd) = self.solve_modified_fourier(DeltaT)

        #Multiscale scheme-----------------------------
        diffusive = 0
        bal = 0
        DeltaTp = np.zeros_like(DeltaT)
        Supbp = np.zeros_like(self.kappam)
        TBp = np.zeros_like(TB)
        Sup,Supp   = np.zeros((2,len(self.kappam)))
        Supb,Supbp = np.zeros((2,len(self.kappam)))
        J,Jp = np.zeros((2,self.n_elems,3))
        #kappa_bal,kappa_balp = np.zeros((2,self.n_parallel))

        for n,q in enumerate(self.rr):
           #COMPUTE BALLISTIC----------------------- 
           if self.multiscale_ballistic:
              X_bal = self.get_X(-1,n,TB,DeltaT)
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
               X = tf[m] - self.mfp_sampled[m]*np.dot(self.VMFP[q],tfg[m].T)
               diffusive +=1
             else:
               X = self.get_X(m,n,TB,DeltaT)
               kappap[m,q] = -np.dot(self.kappa_mask,X)
               if self.multiscale:
                if abs(kappap[m,q] - kappaf[m,q])/abs(kappap[m,q]) < self.error_multiscale :
                  kappap[m,q] = kappaf[m,q]
                  diffusive +=1
                  fourier=True

             #-------------------------------------     
             DeltaTp += X*self.tc[m,q]
             for c,i in enumerate(self.eb): TBp[m,c] -= X[i]*self.GG[m,q,c]
             Jp += np.einsum('c,j->cj',X,self.sigma[m,q])*1e-18
             Supp += kappap[m,q]*self.suppression[m,q,:,0]*self.kappa_factor*1e-9
 
           ballistic = False
           for m in range(self.n_serial)[idx[0]+1:]:
              if ballistic:
               kappap[m,q] = kappa_bal
               X = X_bal
               bal +=1
              else: 
               X = self.get_X(m,n,TB,DeltaT)
               kappap[m,q] = -np.dot(self.kappa_mask,X)
               if abs(kappap[m,q] - kappa_bal)/abs(kappap[m,q]) < self.error_multiscale :
                   kappap[m,q] = kappa_bal
                   bal +=1
                   ballistic=True

              DeltaTp += X*self.tc[m,q]
              for c,i in enumerate(self.eb): TBp[m,c] -= X[i]*self.GG[m,q,c]
              Jp += np.einsum('c,j->cj',X,self.sigma[m,q])*1e-18
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

        error = abs(kappa_old-kappa_tot[0])/abs(kappa_tot[0])
        kappa_old = kappa_tot[0]
        kappa_vec.append(kappa_tot[0])
        if self.verbose and comm.rank == 0:   
         print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa_vec[-1],error))

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
      print(colored(' -----------------------------------------------------------','green'))

     if self.multiscale and comm.rank == 0:
        print()
        print('                  Multiscale Diagnostics        ''')
        print(colored(' -----------------------------------------------------------','green'))

        diff = int(MM[0])/self.n_serial/self.n_parallel 
        bal = int(MM[1])/self.n_serial/self.n_parallel 
        print(colored(' BTE:              ','green') + str(round((1-diff-bal)*100,2)) + ' %' )
        print(colored(' FOURIER:          ','green') + str(round(diff*100,2)) + ' %' )
        print(colored(' BALLISTIC:        ','green') + str(round(bal*100,2)) + ' %' )
        print(colored(' Full termination: ','green') + str(termination) )
        print(colored(' -----------------------------------------------------------','green'))

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
     return {'kappa_vec':kappa_vec,'temperature':DeltaT,'flux':J}#,'suppression':Sup)
     #,'suppression_ballistic':Supb,'suppression_diffusive':Supd}


  def solve_bte(self,**argv):

     if comm.rank == 0:
           print()
           print('      Iter    Thermal Conductivity [W/m/K]      Error ''')
           print(colored(' -----------------------------------------------------------','green'))   


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

      TBp = np.zeros_like(TB)
      Xp = np.zeros_like(X)
      J,Jp = np.zeros((2,self.n_elems,3))
      kappa,kappap = np.zeros((2,self.n_parallel))
      for n,q in enumerate(self.rr):
         Xp[q] = self.get_X_full(n,TB,DeltaT)
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
        print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa_vec[-1],error))
      kk+=1

     if self.verbose and comm.rank == 0:
      print(colored(' -----------------------------------------------------------','green'))

     
     #T = np.einsum('qc,q->c',Xs,self.tc)
     #J = np.einsum('qj,qc->cj',self.sigma,Xs)*1e-18
     T = np.einsum('qc,q->c',X,self.tc)
     J = np.einsum('qj,qc->cj',self.sigma,X)*1e-18
     return {'kappa_vec':kappa_vec,'temperature':T,'flux':J}



  def get_decomposed_directions(self,ll,rot=np.eye(3)):

    normal = self.face_normals[ll]
    dist   = self.dists[ll]
    v_orth = np.dot(normal,np.dot(rot,normal))/np.dot(normal,dist)
    v_non_orth = np.dot(rot,normal) - dist*v_orth
    '''
    if ll == 200:


     elems= self.mesh['side_elem_map_vec'][ll]
     p1 = self.mesh['nodes'][self.mesh['elems'][elems[0]][0]]
     p2 = self.mesh['nodes'][self.mesh['elems'][elems[0]][1]]
     p3 = self.mesh['nodes'][self.mesh['elems'][elems[0]][2]]
     plot([p1[0],p2[0]],[p1[1],p2[1]],'r')
     plot([p2[0],p3[0]],[p2[1],p3[1]],'r')
     plot([p3[0],p1[0]],[p3[1],p1[1]],'r')

     p1 = self.mesh['nodes'][self.mesh['elems'][elems[1]][0]]
     p2 = self.mesh['nodes'][self.mesh['elems'][elems[1]][1]]
     p3 = self.mesh['nodes'][self.mesh['elems'][elems[1]][2]]
     plot([p1[0],p2[0]],[p1[1],p2[1]],'g')
     plot([p2[0],p3[0]],[p2[1],p3[1]],'g')
     plot([p3[0],p1[0]],[p3[1],p1[1]],'g')

     p1 = self.mesh['nodes'][self.mesh['sides'][ll]][0]
     p2 = self.mesh['nodes'][self.mesh['sides'][ll]][1]
     plot([p1[0],p2[0]],[p1[1],p2[1]],'b')

     c1 = self.mesh['centroids'][elems[0]]
     scatter(c1[0],c1[1])
     c1 = self.mesh['centroids'][elems[1]]
     scatter(c1[0],c1[1])

     ss = self.mesh['side_centroids'][ll]
     plot([ss[0],ss[0]+v_non_orth[0]],[ss[1],ss[1]+v_non_orth[1]],'k')
     plot([ss[0],ss[0]+normal[0]],[ss[1],ss[1]+normal[1]],'k')
     
     print(v_non_orth)
     print(normal)
     print(self.mesh['interp_weigths'][ll])

     #show()
     '''

    return v_orth,v_non_orth

  def get_kappa(self,i,j,ll,kappa):

   if i ==j:
    return np.array(kappa[i])
   
   #normal = self.mesh['normals'][i][j]
   normal = self.face_normals[ll]

   kappa_i = np.array(kappa[i])
   kappa_j = np.array(kappa[j])

   ki = np.dot(normal,np.dot(kappa_i,normal))
   kj = np.dot(normal,np.dot(kappa_j,normal))
   w  = self.interp_weigths[ll]

   kappa_loc = kj*kappa_i/(ki*(1-w) + kj*w)
 
   return kappa_loc

   

  def solve_fourier(self,kappa,**argv):

   if np.isscalar(kappa):
       kappa = np.diag(np.diag(kappa*np.eye(3)))

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
   grad = np.zeros((self.n_elems,3))
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
        grad = self.compute_grad(temp,grad)
        C = self.compute_non_orth_contribution(grad,kappa)
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
    #else: 
    # kk = list(self.boundary_sides).index(ll)   
     #print(kk,self.bconn[kk],self.side_centroids[ll]-c1)
    # diff_temp[kc1][ind1] = np.dot(gradT[kc1],self.side_centroids[ll]-c1) 
     #diff_temp[kc1][ind1] = np.dot(gradT[kc1],self.bconn[kk]) 

   gradT = np.zeros((self.n_elems,3))
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

      area = self.areas[ll]   
      w = self.interp_weigths[ll]
      F_ave = w*np.dot(gradT[i],kappa[i]) + (1.0-w)*np.dot(gradT[j],kappa[j])
      #grad_ave = w*gradT[i] + (1.0-w)*gradT[j]

      (dummy,v_non_orth) = self.get_decomposed_directions(ll)#,rot=self.mat['kappa'])

      C[i] += np.dot(F_ave,v_non_orth)/self.volumes[i]*area
      C[j] -= np.dot(F_ave,v_non_orth)/self.volumes[j]*area



    return C


  def compute_diffusive_thermal_conductivity(self,temp,gradT,kappa):

   kappa_eff = 0
   for l in self.flux_sides:

    (i,j) = self.side_elem_map_vec[l]
    (v_orth,v_non_orth) = self.get_decomposed_directions(l,rot=self.get_kappa(i,j,l,kappa))

    deltaT = temp[i] - (temp[j] + 1)
    kappa_eff -= v_orth *  deltaT * self.areas[l]
    w  = self.interp_weigths[l]
    grad_ave = w*gradT[i] + (1.0-w)*gradT[j]
    kappa_eff += np.dot(grad_ave,v_non_orth)/2 * self.areas[l]


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

