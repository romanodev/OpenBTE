from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
import deepdish as dd
from mpi4py import MPI
import scipy.sparse as sp
import time
from matplotlib.pylab import *
import scikits.umfpack as um



comm = MPI.COMM_WORLD



class Solver(object):

  def __init__(self,**argv):

        #COMMON OPTIONS------------
        self.data = argv
        self.tt = np.float64
        self.state = {}
        self.multiscale = argv.setdefault('multiscale',False)
        self.keep_lu = argv.setdefault('keep_lu',True)
        self.verbose = argv.setdefault('verbose',True)
        self.alpha = argv.setdefault('alpha',1.0)
        self.only_fourier = argv.setdefault('only_fourier',False)
        self.max_bte_iter = argv.setdefault('max_bte_iter',20)
        self.max_bte_error = argv.setdefault('max_bte_error',1e-3)
        self.max_fourier_iter = argv.setdefault('max_fourier_iter',20)
        self.max_fourier_error = argv.setdefault('max_fourier_error',1e-5)
        #----------------------------
         
        if comm.rank == 0:
         if self.verbose: self.print_logo()
         print('                         SYSTEM                 ')   
         print(colored(' -----------------------------------------------------------','green'))
         self.print_options()

        #-----IMPORT MESH--------------------------------------------------------------
         if 'geometry' in argv.keys():
          self.mesh = argv['geometry'].data
         else: 
          self.mesh = dd.io.load('geometry.h5')
         #-----------------
         self.i  = self.mesh['i']
         self.j  = self.mesh['j']
         self.k = self.mesh['k']
         self.db = self.mesh['db']
         self.eb = self.mesh['eb']
         self.kappa_mask = self.mesh['kappa_mask']
         self.weigths = self.mesh['weigths_vec']
         self.interp_weigths = self.mesh['interp_weigths']
         self.pp = self.mesh['pp']
         self.meta = self.mesh['meta']
         self.flux_sides = self.mesh['flux_sides']
         self.volumes = self.mesh['volumes']
         self.centroids = self.mesh['centroids']
         self.areas = self.mesh['areas']
         self.dists = self.mesh['dists']
         self.side_elem_map_vec = self.mesh['side_elem_map_vec']
         self.elem_side_map_vec = self.mesh['elem_side_map_vec']
         self.active_sides = self.mesh['active_sides']
         self.face_normals = self.mesh['face_normals']
         self.boundary_sides = self.mesh['boundary_sides']
         self.periodic_sides = self.mesh['periodic_sides']
         self.n_side_per_elems = self.mesh['n_side_per_elem']
         self.periodic_side_values = self.mesh['periodic_side_values']
         self.im = np.concatenate((self.mesh['i'],list(np.arange(self.mesh['meta'][0]))))
         self.jm = np.concatenate((self.mesh['j'],list(np.arange(self.mesh['meta'][0]))))
        self.create_shared_memory(['i','j','im','jm','db','k','eb','kappa_mask','dists','flux_sides',\
                                   'pp','meta','active_sides','areas','side_elem_map_vec','volumes','periodic_sides','weigths',\
                                    'periodic_side_values','n_side_per_elems','centroids','boundary_sides','elem_side_map_vec','interp_weigths','face_normals'])
        self.n_elems = int(self.meta[0])
        self.kappa_factor = self.meta[1]
        self.dim = int(self.meta[2])
        if comm.rank == 0 and self.verbose: self.mesh_info()
        #------------------------------------------------------------------------------
    

        #IMPORT MATERIAL-----------------------------------------
        if comm.rank == 0:
          self.mat = dd.io.load('material.h5')
          self.kappa = self.mat['kappa']
          if not self.only_fourier:
           self.tc = np.array(self.mat['temp'])
           self.ddd = self.tc
           self.sigma = self.mat['G']*1e9
           self.VMFP = self.mat['F']*1e9 
           self.BM = np.zeros(1)
           if self.tc.ndim == 1:
            self.coll = True
            self.meta = np.array([1,len(self.tc),1])
            self.n_parallel = len(self.tc)
            self.tc = np.array([self.tc])
            B = self.mat['B']
            B += B.T - 0.5*np.diag(np.diag(B))
            self.BM = np.einsum('i,ij->ij',self.mat['scale'],B)
            self.sigma = np.array([self.sigma])
            self.VMFP = np.array([self.VMFP]) 
            self.mfp_average = np.zeros(1)
            self.bulk_info()
            self.full_info() 
           else: 
            self.n_serial = self.tc.shape[0]  
            self.n_parallel = self.tc.shape[1]  
            self.meta = np.array([self.n_serial,self.n_parallel,0])
            self.mfp_average = self.mat['mfp_average']
            self.bulk_info()
            self.mfp_info()

        else: self.ddd = None
        self.ddd = comm.bcast(self.ddd,root=0)
        self.create_shared_memory(['sigma','VMFP','tc','meta','BM','mfp_average','kappa'])
        self.n_serial = self.meta[0]           
        self.n_parallel = self.meta[1]           
        self.coll = bool(self.meta[2])
        #-----------------------------------------------------

        if comm.rank == 0:
            print(colored(' -----------------------------------------------------------','green'))
            print(" ")

        #-------SOLVE FOURIER----------------------------------------------
        if comm.rank == 0:
          data = self.solve_fourier(self.kappa)
          variables = {0:{'name':'Temperature Fourier','units':'K',        'data':data['temperature']},\
                     1:{'name':'Flux Fourier'       ,'units':'W/m/m/K','data':data['flux']}}
          self.state.update({'variables':variables,\
                           'kappa_fourier':data['kappa']})
          self.temperature_fourier = data['temperature']
          self.kappa_fourier = np.array([data['kappa']])
          self.fourier_error = np.array([data['error']])
          self.fourier_iter  = np.array([data['n_iter']])
          self.fourier_info()
        else: self.temperature_fourier = None
        self.temperature_fourier = comm.bcast(self.temperature_fourier,root=0)

        self.create_shared_memory(['kappa_fourier'])
        #----------------------------------------------------------------
       

        #-------SET Parallel info----------------------------------------------
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
        #-------------------------------

        #SOLVE BTE-------
        if self.coll:    
          data = self.solve_bte(**argv)
        else:
          data = self.solve_mfp(**argv)
        #-----

        #Saving-----------------------------------------------------------------------
        if comm.rank == 0:
          variables = self.state['variables']

          variables[2]    = {'name':'Temperature BTE','units':'K'             ,'data':data['temperature']}
          variables[3]    = {'name':'Flux BTE'       ,'units':'W/m/m/K'       ,'data':data['flux']}

          self.state.update({'variables':variables,\
                           'kappa':data['kappa_vec']})

          if argv.setdefault('save',True):
           dd.io.save('solver.h5',self.state)
          if self.verbose:
           print(' ')   
           print(colored('                 OpenBTE ended successfully','green'))
           print(' ')  



  def fourier_info(self):

          print('                        FOURIER                 ')   
          print(colored(' -----------------------------------------------------------','green'))
          print(colored('  Iterations:                              ','green') + str(self.fourier_iter[0]))
          print(colored('  Relative error:                          ','green') + '%.1E' % (self.fourier_error[0]))
          print(colored('  Fourier Thermal Conductivity [W/m/K]:    ','green') + str(round(self.kappa_fourier[0],3)))
          print(colored(' -----------------------------------------------------------','green'))
          print(" ")


  def print_options(self):
          print(colored('  Multiscale:                              ','green')+ str(self.multiscale))
          print(colored('  Keep Lu:                                 ','green')+ str(self.keep_lu))
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
          print(colored('  Bulk Thermal Conductivity [W/m/K]:       ','green')+ str(round(self.kappa[0,0],4)))


  def mesh_info(self):

          #print('                        SPACE GRID                 ')   
          #print(colored(' -----------------------------------------------------------','green'))
          print(colored('  Dimension:                               ','green') + str(self.dim))
          print(colored('  Number of Elements:                      ','green') + str(self.n_elems))
          print(colored('  Number of Sides:                         ','green') + str(len(self.mesh['sides'])))
          print(colored('  Number of Nodes:                         ','green') + str(len(self.mesh['nodes'])))
          #print(colored(' -----------------------------------------------------------','green'))
          #print(" ")



  def create_shared_memory(self,varss):
       for var in varss:
         #--------------------------------------
         if comm.Get_rank() == 0: 
          tmp = eval('self.' + var)
          if tmp.dtype == np.int64:
              data_type = 0
              itemsize = MPI.INT.Get_size() 
          elif tmp.dtype == np.float64:
              data_type = 1
              itemsize = MPI.DOUBLE.Get_size() 
          else:
              print('data type for shared memory not supported')
              quit()
          size = np.prod(tmp.shape)
          nbytes = size * itemsize
          meta = [tmp.shape,data_type,itemsize]
         else: nbytes = 0; meta = None
         meta = comm.bcast(meta,root=0)

         #ALLOCATING MEMORY---------------
         win = MPI.Win.Allocate_shared(nbytes,meta[2], comm=comm) 
         buf,itemsize = win.Shared_query(0)
         assert itemsize == meta[2]
         dt = 'i' if meta[1] == 0 else 'd'
         output = np.ndarray(buffer=buf,dtype=dt,shape=meta[0]) 

         if comm.rank == 0:
             output[:] = tmp  

         exec('self.' + var + '=output')


  def solve_modified_fourier(self,DeltaT):

         kappafp = np.zeros((self.n_serial,self.n_parallel))   
         kappaf = np.zeros((self.n_serial,self.n_parallel))   
         tfp = np.zeros((self.n_serial,self.n_elems))
         tfgp = np.zeros((self.n_serial,self.n_elems,3))
         tf = np.zeros((self.n_serial,self.n_elems))
         tfg = np.zeros((self.n_serial,self.n_elems,3))
         for m in self.ff:
           dataf = self.solve_fourier(self.mfp_average[m]*1e-18,pseudo=DeltaT)
           tfp[m] = dataf['temperature']
           tfgp[m] = dataf['grad']
           for q in range(self.n_parallel): 
            kappafp[m,q] = -np.dot(self.kappa_mask,dataf['temperature'] - np.dot(self.VMFP[m,q],dataf['grad'].T))
         comm.Allreduce([kappafp,MPI.DOUBLE],[kappaf,MPI.DOUBLE],op=MPI.SUM)
         comm.Allreduce([tfp,MPI.DOUBLE],[tf,MPI.DOUBLE],op=MPI.SUM)
         comm.Allreduce([tfgp,MPI.DOUBLE],[tfg,MPI.DOUBLE],op=MPI.SUM)

         return kappaf,tf,tfg 

  def solve_mfp(self,**argv):

     #umfpack = um.UmfpackContext() 
     
     if comm.rank == 0:
           print()
           print('      Iter    Thermal Conductivity [W/m/K]      Error ''')
           print(colored(' -----------------------------------------------------------','green'))   


     if comm.rank == 0:   
      SS = np.zeros(1)
      Gbp = np.zeros(1)
      if len(self.mesh['db']) > 0:
       Gbm2 = np.einsum('mqj,jn->mqn',self.VMFP ,self.mesh['db'],optimize=True).clip(max=0)
       Gb   = np.einsum('mqj,jn->mqn',self.sigma,self.mesh['db'],optimize=True)
       Gbp = Gb.clip(min=0); Gbm = Gb.clip(max=0)
       with np.errstate(divide='ignore', invalid='ignore'):
         #tmp = 1/Gbm.sum(axis=1); tmp[np.isinf(tmp)] = 0
         #SS = np.einsum('mqc,mc->mqc',Gbm2,tmp)
         tmp = 1/Gbm.sum(axis=0).sum(axis=0); tmp[np.isinf(tmp)] = 0
         SS = np.einsum('mqc,c->mqc',Gbm2,tmp)
      #---------------------------------------------------------------
      data1 = {'Gbp':Gbp}
      data2 = {'SS':SS}
     else: data1 = None; data2 = None 
     data1 = comm.bcast(data1,root = 0)
     data2 = comm.bcast(data2,root = 0)
     Gbp = data1['Gbp']
     SS = data2['SS']

     #Main matrix----
     G = np.einsum('mqj,jn->mqn',self.VMFP[:,self.rr],self.k,optimize=True)
     Gp = G.clip(min=0); Gm = G.clip(max=0)
     D = np.ones((self.n_serial,len(self.rr),self.n_elems))
     for n,i in enumerate(self.i): D[:,:,i] += Gp[:,:,n]

     if len(self.db) > 0: #boundary
      Gb = np.einsum('mqj,jn->mqn',self.VMFP[:,self.rr],self.db,optimize=True)
      Gbp2 = Gb.clip(min=0);
      for n,i in enumerate(self.eb): D[:,:,i]  += Gbp2[:,:,n]
     #---------------------
     lu =  {}
     DeltaT = self.temperature_fourier

     kappa_vec = list(self.kappa_fourier)

     kappa_old = kappa_vec[-1]
     error = 1
     kk = 0

     kappa_tot = np.zeros(1)
     MM = np.zeros(2)
     Mp = np.zeros(2)
     kappap = np.zeros((self.n_serial,self.n_parallel))
     kappa = np.zeros((self.n_serial,self.n_parallel))
     termination = True


     Bm = np.zeros((self.n_serial,self.n_parallel,len(self.eb)))   
     if len(self.db) > 0:
        a = time.time() 
        tmp = np.einsum('mrn->mn',Gbp) 
        for n,i in enumerate(self.eb):
          Bm[:,self.rr,n] += DeltaT[i]*np.einsum('m,mq->mq',tmp[:,n],SS[:,self.rr,n],optimize=True)
          #Bm[:,self.rr,n] += DeltaT[i]*np.einsum('l,mq->mq',tmp[:,n],SS[:,self.rr,n],optimize=True)

     #INITIALIZE UMFPACK----
     Master = sp.csc_matrix((np.arange(len(self.im))+1,(self.im,self.jm)),shape=(self.n_elems,self.n_elems),dtype=self.tt)
     conversion = np.asarray(Master.data-1,np.int) 
     #umfpack.symbolic(Master)
     #---------------------

     lu = {}
     while kk < self.data.setdefault('max_bte_iter',100) and error > self.data.setdefault('max_bte_error',1e-2):

        if self.multiscale: 
         (kappaf,tf,tfg) = self.solve_modified_fourier(DeltaT)

        #Multiscale scheme-----------------------------
        diffusive = 0
        ballistic = 0
        DeltaTp = np.zeros_like(DeltaT)
        Jp = np.zeros((self.n_elems,3))
        J = np.zeros((self.n_elems,3))
        Bmp = np.zeros_like(Bm)
        for n,q in enumerate(self.rr): 
 
           #COMPUTE BALLISTIC----------------------- 
           if self.multiscale:
            RHS = np.zeros(self.n_elems)
            for ss,v in self.pp: RHS[self.i[int(ss)]] = -Gm[-1,n,int(ss)]*v
            for c,i in enumerate(self.eb): RHS[i] += Bm[-1,q,c]
            Master.data = np.concatenate((Gm[-1,n],D[-1,n]))[conversion]
            lu_loc = sp.linalg.splu(Master)
            X_bal = lu_loc.solve(RHS+DeltaT)
            kappa_bal = -np.dot(self.kappa_mask,X_bal)
            idx  = np.argwhere(np.diff(np.sign(kappaf[:,q] - kappa_bal*np.ones(self.n_serial)))).flatten()
            if len(idx) == 0: idx = [self.n_serial-1]
           else: idx = [self.n_serial-1]
           #----------------------------------------

           #------------------
           fourier = False
           for m in range(self.n_serial)[idx[0]::-1]:
           #for m in range(self.n_serial)[::-1]:
              
              if fourier:
               kappap[m,q] = kappaf[m,q]
               X = tf[m] - np.dot(self.VMFP[m,q],tfg[m].T)
               diffusive +=1
              else: 
               #reconstruct RHS---
               RHS = DeltaT
               for ss,v in self.pp: RHS[self.i[int(ss)]] -= Gm[m,n,int(ss)]*v
               for c,i in enumerate(self.eb): RHS[i] += Bm[m,q,c]

               if not (m,q) in lu.keys() :
                Master.data = np.concatenate((Gm[m,n],D[m,n]))[conversion]
                lu_loc = sp.linalg.splu(Master)
               if self.keep_lu: lu[(m,q)] = lu_loc
               else: lu_loc = lu[(m,q)]   
               X = lu_loc.solve(RHS)
               #umfpack.numeric(Master)
               #X = umfpack.solve( um.UMFPACK_A, Master, RHS, autoTranspose = False )
               #--------------------------

              kappap[m,q] = -np.dot(self.kappa_mask,X)
              Jp += np.outer(X,self.sigma[m,q])*1e-18
              #------------------------
              DeltaTp += X*self.ddd[m,q]
              if len(self.db) > 0:
                for c,i in enumerate(self.eb):
                  Bmp[m,:,c] += X[i]*Gbp[m,q,c]*SS[m,:,c]
                  #Bmp[:,:,c] += X[i]*Gbp[m,q,c]*SS[:,:,c]

              if self.multiscale:
               if abs(kappap[m,q] - kappaf[m,q])/abs(kappap[m,q]) < 0.015 :
                  kappap[m,q] = kappaf[m,q]
                  diffusive +=1
                  fourier=True
               else:   
                   if self.multiscale and m == 0:
                       termination = False

           #BALLISTIC               
           ballistic = False
           for m in range(self.n_serial)[idx[0]+1:]:
              if ballistic:
               kappap[m,q] = kappa_bal
               X = X_bal
               ballistic +=1
              else: 
               #reconstruct RHS---
               RHS = np.zeros(self.n_elems)
               for ss,v in self.pp: RHS[self.i[int(ss)]] = -Gm[m,n,int(ss)]*v
               for c,i in enumerate(self.eb): RHS[i] += Bm[m,q,c]

               if not (m,q) in lu.keys() :
                Master.data = np.concatenate((Gm[m,n],D[m,n]))[conversion]
                lu_loc = sp.linalg.splu(Master)
                if self.keep_lu: lu[(m,q)] = lu_loc
               else: lu_loc = lu[(m,q)]   
               X = lu_loc.solve(RHS + DeltaT)

               Master.data = np.concatenate((Gm[m,n],D[m,n]))[conversion]
               a = time.time()
               #umfpack.numeric(Master)
               #X = umfpack.solve( um.UMFPACK_A, Master, RHS, autoTranspose = True )
               #if not (m,q) in lu.keys() :
               # Master.data = np.concatenate((Gm[m,n],D[m,n]))[conversion]
               # lu_loc = sp.linalg.splu(Master)
               # lu[(m,q)] = lu_loc
               #else: lu_loc = lu[(m,q)]   
               #X = lu_loc.solve(RHS + DeltaT)
               #--------------------------

               #kappap[m,q] = -np.dot(self.kappa_mask,X)
               #Jp += np.outer(X,self.sigma[m,q])*1e-18
               #------------------------
               #DeltaTp += X*self.ddd[m,q]
               #if len(self.db) > 0:
               # for c,i in enumerate(self.eb):
               #   Bmp[m,:,c] += X[i]*Gbp[m,q,c]*SS[m,:,c]

               if self.multiscale:
                if abs(kappap[m,q] - kappa_bal)/abs(kappap[m,q]) < 0.1 :
                   kappap[m,q] = kappa_bal
                   ballistic +=1
                   ballistic=True

        Mp[0] = diffusive
        Mp[1] = ballistic
        comm.Allreduce([DeltaTp,MPI.DOUBLE],[DeltaT,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Jp,MPI.DOUBLE],[J,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Bmp,MPI.DOUBLE],[Bm,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
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
        bal = int(MM[1])/self.n_serial/self.n_parallel 
        print(colored(' BTE:              ','green') + str(round((1-diff-bal)*100,2)) + ' %' )
        print(colored(' FOURIER:          ','green') + str(round(diff*100,2)) + ' %' )
        print(colored(' BALLISTIC:          ','green') + str(round(bal*100,2)) + ' %' )
        print(colored(' Full termination: ','green') + str(termination) )
        print(colored(' -----------------------------------------------------------','green'))

     return {'kappa_vec':kappa_vec,'temperature':DeltaT,'flux':J}


  def solve_bte(self,**argv):

     if comm.rank == 0:
           print()
           print('      Iter    Thermal Conductivity [W/m/K]      Error ''')
           print(colored(' -----------------------------------------------------------','green'))   

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
     G = np.einsum('mqj,jn->mqn',self.VMFP[:,self.rr],self.k,optimize=True)
     Gp = G.clip(min=0); Gm = G.clip(max=0)

     D = np.ones((self.n_serial,len(self.rr),self.n_elems))

     for n,i in enumerate(self.i): 
         D[:,:,i] += Gp[:,:,n]
     if len(self.db) > 0:
      Gb = np.einsum('mqj,jn->mqn',self.VMFP[:,self.rr],self.db,optimize=True)
      Gbp2 = Gb.clip(min=0);
      for n,i in enumerate(self.eb): D[:,:,i]  += Gbp2[:,:,n]

     lu =  {}

     X = np.tile(self.temperature_fourier,(self.n_serial,self.n_parallel,1))
     X_old = X.copy()
     kappa_vec = list(self.kappa_fourier)
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
     kappa = np.zeros((self.n_serial,self.n_parallel))

     #INITIALIZE UMFPACK----
     Master = sp.csc_matrix((np.arange(len(self.im))+1,(self.im,self.jm)),shape=(self.n_elems,self.n_elems),dtype=self.tt)
     conversion = np.asarray(Master.data-1,np.int) 
     #---------------------


     while kk < self.data.setdefault('max_bte_iter',100) and error > self.data.setdefault('max_bte_error',1e-2):

       kappap = np.zeros((self.n_serial,self.n_parallel))
      # COMMON----- 
       Bm = np.zeros((self.n_serial,len(self.rr),self.n_elems))   
       if len(self.db) > 0: 
         for n,i in enumerate(self.eb):
               Bm[:,:,i] +=np.einsum('mu,mu,lq->lq',X[:,:,i],Gbp[:,:,n],SS[:,self.rr,n],optimize=True)
       
       DeltaT = np.matmul(self.BM[self.rr],alpha*X[0]+(1-alpha)*X_old[0]) 
       for n,i in enumerate(self.rr):

            
            if not i in lu.keys() :
                Master.data = np.concatenate((Gm[0,n],D[0,n]))[conversion]
                lu_loc = sp.linalg.splu(Master)
                lu[i] = lu_loc
            else: lu_loc = lu[i]   

            
            #PERIODIC--
            P = np.zeros(self.n_elems)
            for ss,v in self.pp:  P[self.i[int(ss)]] = -Gm[0,n,int(ss)]*v
            #---------

            Xp[0,i] = lu_loc.solve(DeltaT[n] + Bm[0,n] + P)
            kappap[0,i] -= np.dot(self.kappa_mask,Xp[0,i])

       comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
       comm.Allreduce([Xp,MPI.DOUBLE],[X,MPI.DOUBLE],op=MPI.SUM)

       kappa_totp = np.array([np.einsum('mq,mq->',self.sigma[:,self.rr,0],kappa[:,self.rr])])*self.kappa_factor*1e-18
       comm.Allreduce([kappa_totp,MPI.DOUBLE],[kappa_tot,MPI.DOUBLE],op=MPI.SUM)


       error = abs(kappa_old-kappa_tot[0])/abs(kappa_tot[0])
       kappa_old = kappa_tot[0]
       kappa_vec.append(kappa_tot[0])
       if self.verbose and comm.rank == 0:   
        print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa_vec[-1],error))
       kk+=1

     if self.verbose and comm.rank == 0:
      print(colored(' -----------------------------------------------------------','green'))

     
     T = np.einsum('mqc,mq->c',X,self.tc)
     J = np.einsum('mqj,mqc->cj',self.sigma,X)*1e-18
     return {'kappa_vec':kappa_vec,'temperature':T,'flux':J}



  def get_decomposed_directions(self,i,j,ll,rot=np.eye(3)):

     #normal = self.mesh['normals'][i][j]
     normal = self.face_normals[ll]
     #dist   = self.mesh['dists'][i][j]
     dist   = self.dists[ll]

     v_orth = np.dot(normal,np.dot(rot,normal))/np.dot(normal,dist)
     v_non_orth = np.dot(rot,normal) - dist*v_orth
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

    F = sp.dok_matrix((self.n_elems,self.n_elems))
    B = np.zeros(self.n_elems)

    for ll in self.active_sides:

      area = self.areas[ll] 
      (i,j) = self.side_elem_map_vec[ll]
      vi = self.volumes[i]
      vj = self.volumes[j]
      kappa_loc = self.get_kappa(i,j,ll,kappa)
      if not i == j:
       (v_orth,dummy) = self.get_decomposed_directions(i,j,ll,rot=kappa_loc)
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
      B = B + argv['pseudo']
      scale = 1/F.max(axis=0).toarray()[0]
      F.data = F.data * scale[F.indices]
    else:  
      scale = 1/F.max(axis=0).toarray()[0]
      n = np.random.randint(self.n_elems)
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
        grad = self.compute_grad(temp)
        C = self.compute_non_orth_contribution(grad,kappa)
    flux = -np.einsum('cij,cj->ci',kappa,grad)

    return {'flux':flux,'temperature':temp,'kappa':kappa_eff,'grad':grad,'error':error,'n_iter':n_iter}

  def compute_grad(self,temp):

   diff_temp = self.n_elems*[None]
   for i in range(len(diff_temp)):
      diff_temp[i] = self.n_side_per_elems[i]*[0] 

   gradT = np.zeros((self.n_elems,3))
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
 
   for k in range(self.n_elems) :
    #tmp = np.dot(self.mesh['weigths'][k],diff_temp[k])
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
      #print(self.mesh['interp_weigths'])
      #w  = self.mesh['interp_weigths'][ll][0]
      w = self.interp_weigths[ll]
      #F_ave = w*np.dot(gradT[i],self.mat['kappa']) + (1.0-w)*np.dot(gradT[j],self.mat['kappa'])
      F_ave = w*np.dot(gradT[i],kappa[i]) + (1.0-w)*np.dot(gradT[j],kappa[j])
      grad_ave = w*gradT[i] + (1.0-w)*gradT[j]

      (_,v_non_orth) = self.get_decomposed_directions(i,j,ll)#,rot=self.mat['kappa'])

      C[i] += np.dot(F_ave,v_non_orth)/2.0/self.volumes[i]*area
      C[j] -= np.dot(F_ave,v_non_orth)/2.0/self.volumes[j]*area

    return C


  def compute_diffusive_thermal_conductivity(self,temp,gradT,kappa):

   kappa_eff = 0
   for l in self.flux_sides:

    (i,j) = self.side_elem_map_vec[l]
    #(v_orth,v_non_orth) = self.get_decomposed_directions(i,j,rot=self.mat['kappa'])
    (v_orth,v_non_orth) = self.get_decomposed_directions(i,j,l,rot=self.get_kappa(i,j,l,kappa))

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

