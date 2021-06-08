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


def print_multiscale(n_serial,n_parallel,KM):

        #print(KM)
        total = n_serial*n_parallel
        print(flush=True)
        print('                  Multiscale Diagnostics        ''',flush=True)
        print(colored(' -----------------------------------------------------------','green'),flush=True)
        #diff = int(MM[0])/total
        #bal = int(MM[1])/total
        print(colored(' BTE:              ','green') + str(round(KM[0]/np.sum(KM)*100,2)) + ' %',flush=True)
        print(colored(' FOURIER:          ','green') + str(round(KM[1]/np.sum(KM)*100,2)) + ' %',flush=True)
        print(colored(' BALLISTIC:        ','green') + str(round(KM[2]/np.sum(KM)*100,2)) + ' %',flush=True)
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

    kappa_vec = [fourier['meta'][0]]
    kappa_old = kappa_vec[-1]
    error = 1
    kk = 0
    kappa_tot = np.zeros(1)

    kappa,kappap = np.zeros((2,n_serial,n_parallel))
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

    intermediate = {}
    while kk < argv['max_bte_iter'] and error > argv['max_bte_error']:

        #Multiscale scheme-----------------------------
        diffusive = 0
        bal = 0
        DeltaTp = np.zeros_like(DeltaT)
        TBp = np.zeros_like(TB)
        KP = np.zeros(3)
        KM = np.zeros(3)
        Jp = np.zeros_like(J)
        kappa_bal,kappa_balp = np.zeros((2,n_parallel))

        RHS = -np.einsum('c,nc->nc',TB,Gbm2) if len(mesh['db']) > 0 else  np.zeros((argv['n_parallel'],0)) 

        for n,q in enumerate(rr):
           
           for m in range(n_serial):
            
                 #----------------------------------------
                 B = DeltaT +  mfp[m]*(P[n] + get_boundary(RHS[n],mesh['eb'],n_elems))
                 data = np.concatenate((mfp[m]*Gm[n],mfp[m]*D[n]+np.ones(n_elems)))
                 A = get_m(Master,data[conversion])
                 X = solve(argv,A,B,(q,m)) 
                 kappap[m,q] = np.dot(mesh['kappa_mask'],X)

                 DeltaTp += X*mat['tc'][m,q]

                 if len(mesh['eb']) > 0:
                  np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-X[mesh['eb']]*GG[m,q])

                 Jp += np.einsum('c,j->cj',X,sigma[m,q,0:dim])*1e-18

        DeltaT_old = DeltaT.copy()

        comm.Barrier()

        comm.Allreduce([DeltaTp,MPI.DOUBLE],[DeltaT,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Jp,MPI.DOUBLE],[J,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([TBp,MPI.DOUBLE],[TB,MPI.DOUBLE],op=MPI.SUM)
        kappa_totp = np.array([np.einsum('mq,mq->',sigma[:,rr,int(mesh['meta'][-2])],kappa[:,rr])])
        comm.Allreduce([kappa_totp,MPI.DOUBLE],[kappa_tot,MPI.DOUBLE],op=MPI.SUM)


        kk +=1
        error = abs(kappa_old-kappa_tot[0])/abs(kappa_tot[0])
        kappa_old = kappa_tot[0]
        kappa_vec.append(kappa_tot[0])
        if argv['verbose'] and comm.rank == 0:   
         print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa_vec[-1],error),flush=True)


    if argv['verbose'] and comm.rank == 0:
      print(colored(' -----------------------------------------------------------','green'),flush=True)


    bte =  {'kappa':kappa_vec,'temperature':DeltaT,'flux':J,'kappa_mode':kappa}
    argv['bte'] = bte



          
     








