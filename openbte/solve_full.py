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

comm = MPI.COMM_WORLD

      
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



def solve_full(argv):

    mesh    = argv['geometry']
    mat     = argv['material']
    fourier = argv['fourier']
    n_elems = len(mesh['elems'])

    #Post processing
    sigma = mat['sigma']*1e9
    W     = mat['W']

  
    #print(compute_kappa_iter(W,sigma,mat['alpha'][0]))
    

    #print(compute_kappa(np.diag(np.diag(W)),sigma)/mat['alpha'][0])
    #quit()
    B     = -np.einsum('i,ij->ij',1/np.diag(W),W-np.diag(np.diag(W)))
    Wod   = -(W-np.diag(np.diag(W)))
    H     = np.einsum('i,ij->ij',1/np.diag(W),Wod)
    F     = np.einsum('i,ij->ij',1/np.diag(W),sigma)

    #print(max(np.linalg.norm(F,axis=1)))
    #quit()
    tc = mat['tc']

 #------------------
    n_parallel = len(sigma)
    block =  n_parallel//comm.size
    rr = range(block*comm.rank,n_parallel) if comm.rank == comm.size-1 else range(block*comm.rank,block*(comm.rank+1))


    if comm.rank == 0 and argv['verbose']:
       print(flush=True)
       print('      Iter    Thermal Conductivity [W/m/K]      Error ''',flush=True)
       print(colored(' -----------------------------------------------------------','green'),flush=True)

    if comm.rank == 0:   
      if len(mesh['db']) > 0:
       Gb   = np.einsum('qj,jn->qn',sigma,mesh['db'],optimize=True)
       Gbp2 = Gb.clip(min=0);
       with np.errstate(divide='ignore', invalid='ignore'):
         tot = 1/Gb.clip(max=0).sum(axis=0); tot[np.isinf(tot)] = 0
       data = {'GG': np.einsum('qs,s->qs',Gbp2,tot)}
       del tot, Gbp2
      else: data = {'GG':np.zeros(1)}
    else: data = None
    GG = create_shared_memory_dict(data)['GG']

    #MAIN MATRIX
    G = np.einsum('qj,jn->qn',F[rr],mesh['k'],optimize=True)
    Gp = G.clip(min=0); Gm = G.clip(max=0)
    D = np.zeros((len(rr),n_elems))
    np.add.at(D.T,mesh['i'],Gp.T)

    DeltaT = fourier['temperature'].copy()
    TB = np.zeros(len(mesh['eb']))
    if len(mesh['db']) > 0: #boundary
      tmp = np.einsum('rj,jn->rn',F[rr],mesh['db'],optimize=True) 
      Gbp2 = tmp.clip(min=0); Gbm2 = tmp.clip(max=0);
      np.add.at(D.T,mesh['eb'],Gbp2.T)
      TB = DeltaT[mesh['eb']]
    #Periodic---
    sss = np.asarray(mesh['pp'][:,0],dtype=int)
    P = np.zeros((len(rr),n_elems))
    np.add.at(P.T,mesh['i'][sss],-(mesh['pp'][:,1]*Gm[:,sss]).T)
    del G,Gp
    #----------------------------------------------------
     
    #Shared objects
    Xs = shared_array(np.tile(fourier['temperature'],(n_parallel,1)) if comm.rank == 0 else None)
    DeltaTs = shared_array(np.tile(fourier['temperature'],(n_parallel,1)) if comm.rank == 0 else None)
    Xs_old = shared_array(np.tile(fourier['temperature'],(n_parallel,1)) if comm.rank == 0 else None)
    Xs_old[:,:] = Xs[:,:].copy()
    TB_old = TB.copy()

    
    im = np.concatenate((mesh['i'],list(np.arange(n_elems))))
    jm = np.concatenate((mesh['j'],list(np.arange(n_elems))))
    lu = {} 
    kappa_vec = [fourier['meta'][0]]
    kappa_old = kappa_vec[-1]
    kappa_tot = np.zeros(1)
    kk = 0
    error = 1
    alpha=1
    
    nm = len(rr)
    a = time.time()
    while kk < argv['max_bte_iter'] and error > argv['max_bte_error']:
    
      DeltaTs = np.matmul(H[rr],alpha*Xs + (1-alpha)*Xs_old)

      comm.Barrier()

      Xs_old[rr,:] = Xs[rr,:].copy()
  
      RHS = -np.einsum('c,nc->nc',alpha*TB + (1-alpha)*TB_old,Gbm2) if len(mesh['db']) > 0 else np.zeros(n_elems)

      #RHS_tmp = np.zeros((nm,n_elems))
      #if len(mesh['eb'])  > 0:
      #   np.add.at(RHS_tmp,mesh['eb'],RHS[n])


      TBp = np.zeros_like(TB)
      kappa,kappap = np.zeros((2,n_parallel))

      Master = sp.csc_matrix((np.arange(len(im))+1,(im,jm)),shape=(n_elems,n_elems),dtype=np.float64)
      conversion = np.asarray(Master.data-1,np.int) 

      for n,q in enumerate(rr):

          RHS_tmp = np.zeros(n_elems)
          if len(mesh['eb'])  > 0:
            np.add.at(RHS_tmp,mesh['eb'],RHS[n])

          #CORE-----
          Master.data = np.concatenate((Gm[n],D[n]+np.ones(n_elems)))[conversion]
          B = DeltaTs[n] + P[n] + RHS_tmp
          Xs[q] = (lu[q] if q in lu.keys() else lu.setdefault(q,sp.linalg.splu(Master))).solve(B)

          if len(mesh['eb']) > 0:
           np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-Xs[q,mesh['eb']]*GG[q,:])
          
          kappap[q] += np.dot(mesh['kappa_mask'],Xs[q])

     

      TB_old = TB.copy()
      comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
      comm.Allreduce([TBp,MPI.DOUBLE],[TB,MPI.DOUBLE],op=MPI.SUM)
      kappa_totp = np.array([np.einsum('q,q->',sigma[rr,0],kappa[rr])])/mat['alpha'][0]

      comm.Allreduce([kappa_totp,MPI.DOUBLE],[kappa_tot,MPI.DOUBLE],op=MPI.SUM)
      comm.Barrier()
  
      kk +=1
      error = abs(kappa_old-kappa_tot[0])/abs(kappa_tot[0])
      kappa_old = kappa_tot[0]
      kappa_vec.append(kappa_tot[0])
      if argv['verbose'] and comm.rank == 0:   
        print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa_vec[-1],error),flush=True)


    if argv['verbose'] and comm.rank == 0:
      print(colored(' -----------------------------------------------------------','green'),flush=True)

    comm.Barrier()
    print(time.time()-a)
    T = np.einsum('qc,q->c',Xs,tc)
    J = np.einsum('qj,qc->cj',sigma,Xs)*1e-18

    return {'kappa_vec':kappa_vec,'temperature':T,'flux':J}
















