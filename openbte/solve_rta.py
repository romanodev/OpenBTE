from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
from .fourier import *
from mpi4py import MPI
import scipy.sparse as sp
import time
import sys
import scipy
from cachetools import cached,LRUCache
from cachetools.keys import hashkey

comm = MPI.COMM_WORLD
      
cache_compute_lu = LRUCache(maxsize=10000)

def clear_BTE_cache():

  cache_compute_lu.clear()

      
def get_m(Master,data):
    
    Master.data = data
    return Master

def get_boundary(RHS,eb,n_elems):

    if len(eb)  > 0:
     RHS_tmp = np.zeros(n_elems) 
     np.add.at(RHS_tmp,eb,RHS)
    else: 
      return np.zeros(n_elems)  

    return RHS_tmp


def print_multiscale(n_serial,n_parallel,MM):

        total = n_serial*n_parallel

        print(flush=True)
        print('                  Multiscale Diagnostics        ''',flush=True)
        print(colored(' -----------------------------------------------------------','green'),flush=True)
        diff = int(MM[0])/total
        bal = int(MM[1])/total
        print(colored(' BTE:              ','green') + str(round((1-diff-bal)*100,2)) + ' %',flush=True)
        print(colored(' FOURIER:          ','green') + str(round(diff*100,2)) + ' %',flush=True)
        print(colored(' BALLISTIC:        ','green') + str(round(bal*100,2)) + ' %',flush=True)
        print(colored(' -----------------------------------------------------------','green'),flush=True)


@cached(cache=cache_compute_lu, key=lambda data,indices:hashkey(indices))
def compute_lu(A,indices):

 return sp.linalg.splu(A)




def solve_rta(argv):

    mesh    = argv['geometry']
    mat     = argv['material']
    fourier = argv['fourier']
    n_elems = len(mesh['elems'])
    mfp = mat['mfp_sampled']*1e9
   
    sigma = mat['sigma']*1e9
    dim = int(mesh['meta'][2])
    F = mat['VMFP']
    #Get discretization----
    n_serial,n_parallel = sigma.shape[:2]


    block =  n_parallel//comm.size
    rr = range(block*comm.rank,n_parallel) if comm.rank == comm.size-1 else range(block*comm.rank,block*(comm.rank+1))

    if comm.rank == 0 and argv['verbose']:
       print(flush=True)
       print('      Iter    Thermal Conductivity [W/m/K]      Residual ''',flush=True)
       print(colored(' -----------------------------------------------------------','green'),flush=True)

    if len(mesh['db']) > 0:
      if comm.rank == 0: 
       Gb   = np.einsum('mqj,jn->mqn',mat['sigma'],mesh['db'],optimize=True)
       Gbp2 = Gb.clip(min=0);
       with np.errstate(divide='ignore', invalid='ignore'):
          tot = 1/Gb.clip(max=0).sum(axis=0).sum(axis=0); tot[np.isinf(tot)] = 0
          data = {'GG': np.einsum('mqs,s->mqs',Gbp2,tot)}
       del tot, Gbp2
      else: data = None
      if comm.size > 1:
       GG = create_shared_memory_dict(data)['GG']
      else: 
       GG = data['GG']
 
    #Bulk properties---
    G = np.einsum('qj,jn->qn',F[rr],mesh['k'],optimize=True)
    Gp = G.clip(min=0); Gm = G.clip(max=0)
    D = np.zeros((len(rr),len(mesh['elems'])))
    np.add.at(D.T,mesh['i'],Gp.T)
    DeltaT = fourier['temperature'].copy()
    #--------------------------

    #---boundary----------------------
    if len(mesh['db']) > 0: #boundary
     tmp = np.einsum('rj,jn->rn',mat['VMFP'][rr],mesh['db'],optimize=True)  
     Gbp2 = tmp.clip(min=0); Gbm2 = tmp.clip(max=0);
     np.add.at(D.T,mesh['eb'],Gbp2.T)
     TB = DeltaT[mesh['eb']]

    #Periodic---
    sss = np.asarray(mesh['pp'][:,0],dtype=int)
    P = np.zeros((len(rr),n_elems))
    np.add.at(P.T,mesh['i'][sss],-(mesh['pp'][:,1]*Gm[:,sss]).T)
    del G,Gp
    #----------------------------------------------------

    def sparse_dense_product(i,j,B,X):
     '''
     This solves B_ucc' X_uc' -> A_uc

     B_ucc' : sparse in cc' and dense in u. Data is its vectorized data COO descrition
     X      : dense matrix
     '''

     tmp = np.zeros_like(X)
     np.add.at(tmp.T,i,B.T * X.T[j])   

     return tmp

    def compute_upwind_projected_gradient(X,m,q):

         return sparse_dense_product(mesh['i'],mesh['j'],Gm[q],X) + np.einsum('c,c->c',D[q],X) - P[q]

    def compute_partial_residual(X,m,q,B):

       R = mfp[m]*compute_upwind_projected_gradient(X,m,q) + X - B

       return np.linalg.norm(R)/np.linalg.norm(X)
       

    def compute_residual(X):

       R = np.zeros_like(X)

       DeltaT = np.zeros(n_elems)
       TB = np.zeros(len(mesh['eb']))
       for m in range(n_serial):
         for q in range(n_parallel):
           DeltaT += mat['tc'][m,q]*X[q,m]
           np.add.at(TB,np.arange(mesh['eb'].shape[0]),-X[q,m,mesh['eb']]*GG[m,q])
      
       #Boundary----------
       RHS = -np.einsum('c,nc->cn',TB,Gbm2) 
       B   =  np.zeros((n_elems,n_parallel))
       np.add.at(B,mesh['eb'],RHS)
       #------------------

       for m in range(n_serial):
           R[:,m,:] = (sparse_dense_product(mesh['i'],mesh['j'],Gm,X[:,m,:]) + np.einsum('uc,uc->uc',D,X[:,m,:]) - P - B.T) + (X[:,m,:]-DeltaT)/mfp[m]

       R  = R.reshape((n_serial*n_parallel,n_elems))

       return np.linalg.norm(R,ord='fro')


    kappa_vec = [fourier['meta'][0]]
    kappa_old = kappa_vec[-1]
    error = 1
    kk = 0
    kappa_tot = np.zeros(1)
    kappaf_tot = np.zeros(1)
    MM = np.zeros(2)
    Mp = np.zeros(2)
    kappa,kappap = np.zeros((2,n_serial,n_parallel))
    kappaf,kappap_f = np.zeros((2,n_serial,n_parallel))
    kappa_0,kappap_0 = np.zeros((2,n_serial,n_parallel))
    kappab,kappap_b = np.zeros((2,n_serial,n_parallel))
    transitionp = np.zeros(n_parallel)
    transition = 1e4 * np.ones(n_parallel)
    mat['tc'] = mat['tc']/np.sum(mat['tc'])
    im = np.concatenate((mesh['i'],list(np.arange(n_elems))))
    jm = np.concatenate((mesh['j'],list(np.arange(n_elems))))
    Master = sp.csc_matrix((np.arange(len(im))+1,(im,jm)),shape=(n_elems,n_elems),dtype=np.float64)
    conversion = np.asarray(Master.data-1,np.int)
    J = np.zeros((n_elems,dim))

    tau,taup = np.zeros((2,n_serial,n_parallel,2))
   
    lu = {}

    def solve(argv,A,B,indices):

     if not argv['keep_lu']:
         X = sp.linalg.spsolve(A,B)
     else: 
         
         if indices in lu:
            ll = lu[indices]
         else:   
            ll = sp.linalg.splu(A)
            lu[indices] = ll

         X = ll.solve(B)

     return X 

    gradT = compute_grad(fourier['temperature'],**argv)  

    #a = np.einsum('m,ui,ci->umc',mfp,F,gradT)
    #X_tot  = fourier['temperature'][None,None,...] - a
    #X_tot = np.tile(fourier['temperature'],(n_parallel,n_serial,1))
    #compute_residual(X_tot)

    def compute_anisotropic_tau(X):


      grad = np.zeros((2,n_elems))

      G = np.einsum('qj,jn->qn',[[1.0,0],[0.0,1.0]],mesh['k'],optimize=True)
      Gp = G.clip(min=0); Gm = G.clip(max=0)
      D = np.zeros((2,n_elems))
      np.add.at(D.T,mesh['i'],Gp.T)

      sss = np.asarray(mesh['pp'][:,0],dtype=int)
      P = np.zeros((2,n_elems))
      np.add.at(P.T,mesh['i'][sss],-(mesh['pp'][:,1]*Gm[:,sss]).T)

      grad[0] = sparse_dense_product(mesh['i'],mesh['j'],Gm[0],X) + np.einsum('c,c->c',D[0],X) - P[0]
      grad[1] = sparse_dense_product(mesh['i'],mesh['j'],Gm[1],X) + np.einsum('c,c->c',D[1],X) - P[1]
 

      return -np.einsum('ic,c->i',grad,mesh['kappa_mask'])*1e18


    #quit()

    while kk < argv['max_bte_iter'] and error > argv['max_bte_error']:
        a = time.time()
        if argv['multiscale']: tf,tfg = solve_fourier(mat['mfp_average'],DeltaT,argv)

        #Multiscale scheme-----------------------------
        diffusive = 0
        bal = 0
        DeltaTp = np.zeros_like(DeltaT)
        TBp = np.zeros_like(TB)
        Jp = np.zeros_like(J)
        kappa_bal,kappa_balp = np.zeros((2,n_parallel))

        RHS = -np.einsum('c,nc->nc',TB,Gbm2) if len(mesh['db']) > 0 else  np.zeros((argv['n_parallel'],0)) 

        for n,q in enumerate(rr):
           
           #Get ballistic kappa
           kappap_0[:,q]   =   -np.einsum('c,m,i,ci->m',mesh['kappa_mask'],mfp,mat['VMFP'][q,0:dim],compute_grad(DeltaT,**argv))
           if argv['multiscale']:
              kappap_f[:,q] =    np.einsum('c,mc->m',mesh['kappa_mask'],tf) - np.einsum('c,m,i,mci->m',mesh['kappa_mask'],mfp,mat['VMFP'][q,0:dim],tfg) 
        
              #------------------------------------------------------
              B = DeltaT + mfp[-1]*(P[n] + get_boundary(RHS[n],mesh['eb'],n_elems))
              A = get_m(Master,np.concatenate((mfp[-1]*Gm[n],mfp[-1]*D[n]+np.ones(n_elems)))[conversion])
              X_bal = solve(argv,A,B,(q,-1))
              kappap_b[:,q] = np.dot(mesh['kappa_mask'],X_bal)
              #-----------------------------------------------------

              idx = np.argwhere(np.diff(np.sign(kappap_b[:,q] - kappap_f[:,q]))).flatten()+1
              if len(idx) == 0:
                 idx = n_serial-1
              elif len(idx) > 1:
                 idx = idx[-1] #take the last one to be on the safe side
              else:   
                 idx = idx[0]

              idx = min([idx,n_serial])

           else: idx = n_serial-1
        
           for m in range(n_serial)[idx::-1]:
            
                 #----------------------------------------
                 B = DeltaT +  mfp[m]*(P[n] + get_boundary(RHS[n],mesh['eb'],n_elems))
                 data = np.concatenate((mfp[m]*Gm[n],mfp[m]*D[n]+np.ones(n_elems)))
                 A = get_m(Master,data[conversion])
                 X = solve(argv,A,B,(q,m)) 
                 kappap[m,q] = np.dot(mesh['kappa_mask'],X)

                 #---------------------------------------
                 #taup[m,q] = compute_anisotropic_tau(X)
                 #---------------------------------------

                 #-----------------------------------------
                 if argv['multiscale']:

                  error = abs(kappap[m,q] - kappap_f[m,q])/abs(kappap[m,q])
                  if error < argv['multiscale_error_fourier']:# and m <= transition[q]:
              
                   #mm[q] = m     
                   transitionp[q] = m if argv.setdefault('transition',False) else 1e4
   
                   #Vectorize
                   kappap[:m+1,q] = kappap_f[:m+1,q]
                   diffusive += m+1
                   Xm = tf[:m+1] - np.einsum('m,i,mci->mc',mat['mfp_sampled'][:m+1],mat['VMFP'][q,:dim],tfg[:m+1])
                   DeltaTp += np.einsum('mc,m->c',Xm,mat['tc'][:m+1,q])  
                  
                   if len(mesh['eb']) > 0:
                     np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-np.einsum('mc,mc->c',Xm[:,mesh['eb']],GG[:m+1,q]))
                   
                   Jp += np.einsum('mc,mj->cj',Xm,sigma[:m+1,q,0:dim])*1e-18
                   break

                 DeltaTp += X*mat['tc'][m,q]

                 if len(mesh['eb']) > 0:
                  np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-X[mesh['eb']]*GG[m,q])

                 Jp += np.einsum('c,j->cj',X,sigma[m,q,0:dim])*1e-18
                
 
           for m in range(n_serial)[idx+1:]:

               #----------------------------------------------------
               B = DeltaT +  mfp[m]*(P[n] + get_boundary(RHS[n],mesh['eb'],n_elems))
               A = get_m(Master,np.concatenate((mfp[m]*Gm[n],mfp[m]*D[n]+np.ones(n_elems)))[conversion])
               X = solve(argv,A,B,(q,m))
               kappap[m,q] = np.dot(mesh['kappa_mask'],X)
               #-------------------------------------------------------

               error_bal = abs(kappap[m,q] - kappap_b[m,q])/abs(kappap[m,q])
               if error_bal < argv['multiscale_error_ballistic'] and \
                   abs(kappap[m-1,q] - kappap_b[m-1,q])/abs(kappap[m-1,q]) < argv['multiscale_error_ballistic'] and  m > int(len(mat['mfp_sampled'])/2):

                   kappap[m:,q] = kappap_b[m:,q]
                   bal += n_serial-m
                   DeltaTp += X_bal*np.sum(mat['tc'][m:,q])
                   

                   if len(mesh['eb']) > 0:
                    np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-np.einsum('c,mc->c',X_bal[mesh['eb']],GG[m:,q]))

                   Jp += np.einsum('c,j->cj',X_bal,np.sum(sigma[m:,q,0:dim],axis=0))*1e-18
                   break

               DeltaTp += X*mat['tc'][m,q]
               
               if len(mesh['eb'])>0:
                np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-X[mesh['eb']]*GG[m,q])

               Jp += np.einsum('c,j->cj',X,sigma[m,q,0:dim])*1e-18

        Mp[0] = diffusive
        Mp[1] = bal

        #Centering--
        #DeltaT = DeltaT - (max(DeltaT)+min(DeltaT))/2.0

        DeltaT_old = DeltaT.copy()

        comm.Barrier()

        comm.Allreduce([DeltaTp,MPI.DOUBLE],[DeltaT,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Jp,MPI.DOUBLE],[J,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
        #comm.Allreduce([taup,MPI.DOUBLE],[tau,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappap_b,MPI.DOUBLE],[kappab,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappap_0,MPI.DOUBLE],[kappa_0,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappap_f,MPI.DOUBLE],[kappaf,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Mp,MPI.DOUBLE],[MM,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([transitionp,MPI.DOUBLE],[transition,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([TBp,MPI.DOUBLE],[TB,MPI.DOUBLE],op=MPI.SUM)
        kappa_totp = np.array([np.einsum('mq,mq->',sigma[:,rr,int(mesh['meta'][-1])],kappa[:,rr])])
        kappaf_totp = np.array([np.einsum('mq,mq->',sigma[:,rr,int(mesh['meta'][-1])],kappaf[:,rr])])
        comm.Allreduce([kappa_totp,MPI.DOUBLE],[kappa_tot,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappaf_totp,MPI.DOUBLE],[kappaf_tot,MPI.DOUBLE],op=MPI.SUM)


        if argv['multiscale'] and comm.rank == 0: print_multiscale(n_serial,n_parallel,MM) 

        kk +=1
        error = abs(kappa_old-kappa_tot[0])/abs(kappa_tot[0])
        kappa_old = kappa_tot[0]
        kappa_vec.append(kappa_tot[0])
        if argv['verbose'] and comm.rank == 0:   
         print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa_vec[-1],error),flush=True)
 

    if argv['verbose'] and comm.rank == 0:
      print(colored(' -----------------------------------------------------------','green'),flush=True)


    kappaf -=np.dot(mesh['kappa_mask'],DeltaT_old)
    kappa -=np.dot(mesh['kappa_mask'],DeltaT_old)
    bte =  {'kappa':kappa_vec,'temperature':DeltaT,'flux':J,'kappa_mode':kappa,'kappa_mode_f':kappaf,\
            'kappa_mode_b':kappab-np.dot(mesh['kappa_mask'],DeltaT_old),'kappa_0':kappa_0}
    argv['bte'] = bte



          
     








