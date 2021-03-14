from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
from .fourier import *
#import deepdish as dd
from mpi4py import MPI
import scipy.sparse as sp
import time
import sys
import scipy
from matplotlib.pylab import *
from cachetools import cached,LRUCache
from cachetools.keys import hashkey

from matplotlib.pylab import *
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


def solve_mg(argv):

    
    mesh    = argv['geometry']
    mat     = argv['material']
    fourier = argv['fourier']
    n_elems = len(mesh['elems'])
    mfp = mat['mfp_sampled']*1e9
    sigma = mat['sigma']
    dim = int(mesh['meta'][2])
    F = mat['VMFP']
    
    #Get discretization----
    n_serial = len(mfp)
    n_parallel = sigma.shape[0]

    block =  n_parallel//comm.size
    rr = range(block*comm.rank,n_parallel) if comm.rank == comm.size-1 else range(block*comm.rank,block*(comm.rank+1))

    #kappa_x = 0
    #for index in range(n_parallel):
    #    for m in range(n_serial):
    #      kappa_x += F[index,0]*sigma[index,0]
    
    #------------
    if mesh['db'].shape[1] > 0:
      if comm.rank == 0:
       Gb   = np.einsum('qj,jn->qn',mat['sigma'],mesh['db'],optimize=True)
       Gbp2 = Gb.clip(min=0);
       with np.errstate(divide='ignore', invalid='ignore'):
          tot = 1/Gb.clip(max=0).sum(axis=0); tot[np.isinf(tot)] = 0
          data = {'GG': np.einsum('qn,n->qn',Gbp2,tot)}
       del tot, Gbp2
      else: data = None
      GG = create_shared_memory_dict(data)['GG']
 
    #---Bulk properties---
    G = np.einsum('qj,jn->qn',F[rr],mesh['k'],optimize=True)
    Gp = G.clip(min=0); Gm = G.clip(max=0)
    D = np.zeros((len(rr),len(mesh['elems'])))
    np.add.at(D.T,mesh['i'],Gp.T)
    DeltaT = np.tile(fourier['temperature'],(n_serial,1))
    #--------------------------

    #---boundary----------------------
    if len(mesh['db']) > 0: #boundary
     tmp = np.einsum('rj,jn->rn',mat['VMFP'][rr],mesh['db'],optimize=True)  
     Gbp2 = tmp.clip(min=0); Gbm2 = tmp.clip(max=0);
     np.add.at(D.T,mesh['eb'],Gbp2.T)
     TB = DeltaT.T[mesh['eb']].T

    if comm.rank == 0:
     print(flush=True)
     print('      Iter            Error ''',flush=True)
     print(colored(' -----------------------------------------------------------','green'),flush=True)

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
    kappaf_tot = np.zeros(1)
    MM = np.zeros(2)
    Mp = np.zeros(2)
    kappa,kappap = np.zeros((2,n_serial,n_parallel))
    mat['tc'] = mat['tc']/np.sum(mat['tc'])
    im = np.concatenate((mesh['i'],list(np.arange(n_elems))))
    jm = np.concatenate((mesh['j'],list(np.arange(n_elems))))
    Master = sp.csc_matrix((np.arange(len(im))+1,(im,jm)),shape=(n_elems,n_elems),dtype=np.float64)
    conversion = np.asarray(Master.data-1,np.int)
    J = np.zeros((n_elems,dim))
    S = np.zeros((n_serial,n_parallel))
    Sp = np.zeros((n_serial,n_parallel))
   
    S_old = S.copy()

    lu = {}

    def solve(argv,A,B,indices):

     if not argv['keep_lu']:
         X = sp.linalg.spsolve(A,B,use_umfpack=True)
     else: 
         if indices in lu:
            ll = lu[indices]
         else:   
            ll = sp.linalg.splu(A)
            lu[indices] = ll
         X = ll.solve(B)
     return X 

    gradT = compute_grad(fourier['temperature'],**argv)  

    while kk < argv['max_bte_iter'] and error > argv['max_bte_error']:

        DeltaTp = np.zeros_like(DeltaT)
        TBp = np.zeros_like(TB)

        RHS = -np.einsum('mc,nc->mnc',TB,Gbm2) if len(mesh['db']) > 0 else  np.zeros((argv['n_parallel'],0)) 

        for n,q in enumerate(rr):
            for m in range(n_serial):
            
                 B = DeltaT[m] + mfp[m]*P[n] + get_boundary(RHS[m,n],mesh['eb'],n_elems)
                 
                 data = np.concatenate((mfp[m]*Gm[n],mfp[m]*D[n]+np.ones(n_elems)))
                 A = get_m(Master,data[conversion])
                 X = solve(argv,A,B,(q,m))
                 Sp[m,q] = np.dot(mesh['kappa_mask'],X-DeltaT[m])*1e18/mfp[m]*sigma[q,0]
                 
                 DeltaTp[m] += X*mat['tc'][q]
                 #-----------------------------------------
                 if len(mesh['eb']) > 0:
                  #X = np.ones_like(X)   
                   np.add.at(TBp[m],np.arange(mesh['eb'].shape[0]),-X[mesh['eb']]*GG[q])



        comm.Barrier()
        comm.Allreduce([DeltaTp,MPI.DOUBLE],[DeltaT,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Sp,MPI.DOUBLE],[S,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([TBp,MPI.DOUBLE],[TB,MPI.DOUBLE],op=MPI.SUM)
        #print(np.min(TB),np.max(TB))
        error = np.linalg.norm(S-S_old,ord='fro')
        Sm = np.sum(S,axis=1)  
        if argv['verbose'] and comm.rank == 0:   
         print('{0:8d}{1:22.4E}'.format(kk,error),flush=True)
         print(Sm)
        S_old = S.copy() 
        kk +=1

    bte =  {'kappa':[0],'temperature':DeltaT[0],'flux':np.zeros((n_elems,2)),'kappa_mode':kappa,'kappa_mode_f':[],\
            'kappa_mode_b':[0],'kappa_0':[0]}
    argv['bte'] = bte

    
    if comm.rank == 0:
     save_data('suppression',{'S':Sm,'mfp':mfp})



          
     








