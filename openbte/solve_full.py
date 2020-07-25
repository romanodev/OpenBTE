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

    #for c,i in enumerate(eb): RHS_tmp[i] += RHS[c]

    return RHS_tmp


def print_multiscale(argv,MM):

        total = argv['n_serial']*argv['n_parallel']

        print(flush=True)
        print('                  Multiscale Diagnostics        ''',flush=True)
        print(colored(' -----------------------------------------------------------','green'),flush=True)
        diff = int(MM[0])/total
        bal = int(MM[1])/total
        print(colored(' BTE:              ','green') + str(round((1-diff-bal)*100,2)) + ' %',flush=True)
        print(colored(' FOURIER:          ','green') + str(round(diff*100,2)) + ' %',flush=True)
        print(colored(' BALLISTIC:        ','green') + str(round(bal*100,2)) + ' %',flush=True)
        print(colored(' -----------------------------------------------------------','green'),flush=True)



def solve_full(**argv):

    mesh    = argv['mesh']
    mat     = argv['mat']
    fourier = argv['fourier']
    n_elems = len(mesh['elems'])

    if comm.rank == 0 and argv['verbose']:
       print(flush=True)
       print('      Iter    Thermal Conductivity [W/m/K]      Error ''',flush=True)
       print(colored(' -----------------------------------------------------------','green'),flush=True)

    if comm.rank == 0:   
      if len(mesh['db']) > 0:
       Gb   = np.einsum('qj,jn->qn',mat['sigma'],mesh['db'],optimize=True)
       Gbp2 = Gb.clip(min=0);
       with np.errstate(divide='ignore', invalid='ignore'):
         tot = 1/Gb.clip(max=0).sum(axis=0); tot[np.isinf(tot)] = 0
       data = {'GG': np.einsum('qs,s->qs',Gbp2,tot)}
       del tot, Gbp2
      else: data = {'GG':np.zeros(1)}
    else: data = None
    GG = create_shared_memory_dict(data)['GG']

    #MAIN MATRIX
    G = np.einsum('qj,jn->qn',mat['VMFP'][argv['rr']],mesh['k'],optimize=True)
    Gp = G.clip(min=0); Gm = G.clip(max=0)
    D = np.zeros((len(argv['rr']),argv['n_elems']))
    np.add.at(D.T,mesh['i'],Gp.T)

    DeltaT = fourier['temperature_fourier'].copy()
    TB = np.zeros(len(mesh['eb']))
    if len(mesh['db']) > 0: #boundary
      tmp = np.einsum('rj,jn->rn',mat['VMFP'][argv['rr']],mesh['db'],optimize=True) 
      Gbp2 = tmp.clip(min=0); Gbm2 = tmp.clip(max=0);
      np.add.at(D.T,mesh['eb'],Gbp2.T)
      TB = DeltaT[mesh['eb']]
    #Periodic---
    sss = np.asarray(mesh['pp'][:,0],dtype=int)
    P = np.zeros((len(argv['rr']),n_elems))
    np.add.at(P.T,mesh['i'][sss],-(mesh['pp'][:,1]*Gm[:,sss]).T)
    del G,Gp
    #----------------------------------------------------
     
    #Shared objects
    Xs = shared_array(np.tile(fourier['temperature_fourier'],(argv['n_parallel'],1)) if comm.rank == 0 else None)
    DeltaTs = shared_array(np.tile(fourier['temperature_fourier'],(argv['n_parallel'],1)) if comm.rank == 0 else None)
    Xs_old = shared_array(np.tile(fourier['temperature_fourier'],(argv['n_parallel'],1)) if comm.rank == 0 else None)
    Xs_old[:,:] = Xs[:,:].copy()
    TB_old = TB.copy()

    lu = {} 
    kappa_vec = [fourier['meta'][0]]
    kappa_old = kappa_vec[-1]
    kappa_tot = np.zeros(1)
    kk = 0
    error = 1
    while kk < argv['max_bte_iter'] and error > argv['max_bte_error']:
       
      DeltaTs = np.matmul(mat['B'][argv['rr']],argv['alpha']*Xs+(1-argv['alpha'])*Xs_old)
   
      RHS = -np.einsum('c,nc->nc',argv['alpha']*TB + (1-argv['alpha'])*TB_old,Gbm2) if len(mesh['db']) > 0 else np.zeros(n_elems)
      
      TBp = np.zeros_like(TB)
      kappa,kappap = np.zeros((2,argv['n_parallel']))

      Master = sp.csc_matrix((np.arange(len(mesh['im']))+1,(mesh['im'],mesh['jm'])),shape=(n_elems,n_elems),dtype=np.float64)
      conversion = np.asarray(Master.data-1,np.int) 

      for n,q in enumerate(argv['rr']):
          
          if len(mesh['eb'])  > 0:
           RHS_tmp = np.zeros(n_elems) 
           np.add.at(RHS_tmp,mesh['eb'],RHS[n])
          else: 
           RHS_tmp =  np.zeros(n_elems)  

          #CORE-----
          Master.data = np.concatenate((Gm[n],D[n]+np.ones(n_elems)))[conversion]
          B = DeltaTs[n] + P[n] + RHS_tmp 
          if argv['keep_lu']:
            Xs[q] = (lu[q] if q in lu.keys() else lu.setdefault(q,sp.linalg.splu(Master))).solve(B)
          else:  
            Xs[q] = sp.linalg.spsolve(Master,B,use_umfpack=True)  
          #---------

          if len(mesh['eb']) > 0:
           np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-Xs[q,mesh['eb']]*GG[q,:])
          
          kappap[q] += np.dot(mesh['kappa_mask'],Xs[q])


     
      Xs_old[:,:] = Xs[:,:].copy()

      TB_old = TB.copy()
      comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
      comm.Allreduce([TBp,MPI.DOUBLE],[TB,MPI.DOUBLE],op=MPI.SUM)
      kappa_totp = np.array([np.einsum('q,q->',mat['sigma'][argv['rr'],0],kappa[argv['rr']])])/mat['alpha'][0]
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

    T = np.einsum('qc,q->c',Xs,mat['tc'])
    J = np.einsum('qj,qc->cj',mat['sigma'],Xs)*1e-18/mat['alpha'][0]

    return {'kappa_vec':kappa_vec,'temperature':T,'flux':J}
















