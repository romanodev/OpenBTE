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
from cachetools import cached
from cachetools.keys import hashkey
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


@cached(cache={}, key=lambda data,indices:hashkey(indices))
def compute_lu(A,indices):

 return sp.linalg.splu(A)

@cached(cache={}, key=lambda data,indices:hashkey(indices))
def compute_spilu(A,indices):

 return sp.linalg.spilu(A)



def solve(argv,A,B,indices):

    if not argv['keep_lu']:
        X = sp.linalg.spsolve(A,B,use_umfpack=True)
    else: 
        X = compute_lu(A,indices).solve(B)

    return X 



def solve_rta(**argv):

    mesh    = argv['mesh']
    mat     = argv['mat']
    fourier = argv['fourier']
    n_elems = len(mesh['elems'])
    mfp = mat['mfp_sampled']*1e9
    sigma = mat['sigma']*1e9

    if comm.rank == 0 and argv['verbose']:
       print(flush=True)
       print('      Iter    Thermal Conductivity [W/m/K]      Error ''',flush=True)
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
      GG = create_shared_memory_dict(data)['GG']
 
    #Bulk properties---
    G = np.einsum('qj,jn->qn',mat['VMFP'][argv['rr']],mesh['k'],optimize=True)
    Gp = G.clip(min=0); Gm = G.clip(max=0)
    D = np.zeros((len(argv['rr']),len(mesh['elems'])))

    #for n,i in enumerate(mesh['i']):  D[:,i] += Gp[:,n]
    np.add.at(D.T,mesh['i'],Gp.T)

    DeltaT = fourier['temperature_fourier'].copy()
    #DeltaT_initial = DeltaT.copy()
    #--------------------------

    #---boundary----------------------
    TB = np.zeros((argv['n_serial'],len(mesh['eb'])))   
    if len(mesh['db']) > 0: #boundary
     tmp = np.einsum('rj,jn->rn',mat['VMFP'][argv['rr']],mesh['db'],optimize=True)  
     Gbp2 = tmp.clip(min=0); Gbm2 = tmp.clip(max=0);

     np.add.at(D.T,mesh['eb'],Gbp2.T)
     TB = DeltaT[mesh['eb']]

    #Periodic---
    sss = np.asarray(mesh['pp'][:,0],dtype=int)
    P = np.zeros((len(argv['rr']),n_elems))
    #for (_,v),ss in zip(mesh['pp'],sss): P[:,mesh['i'][ss]] -= Gm[:,ss].copy()*v
    np.add.at(P.T,mesh['i'][sss],-(mesh['pp'][:,1]*Gm[:,sss]).T)
    del G,Gp
    #----------------------------------------------------

    lu =  {}
    kappa_vec = [fourier['meta'][0]]
    kappa_old = kappa_vec[-1]
    error = 1
    kk = 0
    kappa_tot = np.zeros(1)
    kappaf_tot = np.zeros(1)
    MM = np.zeros(2)
    Mp = np.zeros(2)
    kappa,kappap = np.zeros((2,argv['n_serial'],argv['n_parallel']))
    kappaf,kappap_f = np.zeros((2,argv['n_serial'],argv['n_parallel']))
    kappa0,kappap_0 = np.zeros((2,argv['n_serial'],argv['n_parallel']))
    kappab,kappap_b = np.zeros((2,argv['n_serial'],argv['n_parallel']))
    transitionp = np.zeros(argv['n_parallel'])
    transition = 1e4 * np.ones(argv['n_parallel'])

    mat['tc'] = mat['tc']/np.sum(mat['tc'])

    Master = sp.csc_matrix((np.arange(len(mesh['im']))+1,(mesh['im'],mesh['jm'])),shape=(n_elems,n_elems),dtype=np.float64)
    conversion = np.asarray(Master.data-1,np.int) 

    J = np.zeros((n_elems,argv['dim']))

    a = time.time()
    while kk < argv['max_bte_iter'] and error > argv['max_bte_error']:
        if argv['multiscale']: tf,tfg = solve_fourier(mat['mfp_average'],DeltaT,argv)
        #Multiscale scheme-----------------------------
        diffusive = 0
        bal = 0
        DeltaTp = np.zeros_like(DeltaT)
        TBp = np.zeros_like(TB)
        Jp = np.zeros_like(J)
        kappa_bal,kappa_balp = np.zeros((2,argv['n_parallel']))

        RHS = -np.einsum('c,nc->nc',TB,Gbm2) if len(mesh['db']) > 0 else  np.zeros((argv['n_parallel'],0)) 

        for n,q in enumerate(argv['rr']):
           
           #Get ballistic kappa
           if argv['multiscale']:
              kappap_f[:,q] = np.einsum('c,mc->m',mesh['kappa_mask'],tf) - np.einsum('c,m,i,mci->m',mesh['kappa_mask'],mfp,mat['VMFP'][q,0:argv['dim']],tfg)

              #------------------------------------------------------
              B = DeltaT +  mfp[-1]*(P[n] + get_boundary(RHS[n],mesh['eb'],n_elems))
              A = get_m(Master,np.concatenate((mfp[-1]*Gm[n],mfp[-1]*D[n]+np.ones(n_elems)))[conversion])
              X_bal = solve(argv,A,B,(q,-1))
              kappap_b[:,q] = np.dot(mesh['kappa_mask'],X_bal)
              #-----------------------------------------------------

              idx = np.argwhere(np.diff(np.sign(kappap_b[:,q] - kappap_f[:,q]))).flatten()+1
              if len(idx) == 0:
                 idx = argv['n_serial']-1
              elif len(idx) > 1:
                 idx = idx[-1] #take the last one to be on the safe side
              else:   
                 idx = idx[0]

              idx = min([idx,argv['n_serial']])

           else: idx = argv['n_serial']-1

           for m in range(argv['n_serial'])[idx::-1]:
               
                 #-------------------------
                 B = DeltaT +  mfp[m]*(P[n] +get_boundary(RHS[n],mesh['eb'],n_elems))
                 A = get_m(Master,np.concatenate((mfp[m]*Gm[n],mfp[m]*D[n]+np.ones(n_elems)))[conversion])
                 X = solve(argv,A,B,(q,m))
                 kappap[m,q] = np.dot(mesh['kappa_mask'],X)
                 #-------------------------
 
                 if argv['multiscale']:
                  error = abs(kappap[m,q] - kappap_f[m,q])/abs(kappap[m,q])
                  if error < argv['multiscale_error_fourier'] and m <= transition[q]:

                   transitionp[q] = m  if argv.setdefault('transition',False) else 1e4
   
                   #Vectorize
                   kappap[:m+1,q] = kappap_f[:m+1,q]
                   diffusive += m+1
                   Xm = tf[:m+1] - np.einsum('m,i,mci->mc',mat['mfp_sampled'][:m+1],mat['VMFP'][q,:argv['dim']],tfg[:m+1])

                   DeltaTp += np.einsum('mc,m->c',Xm,mat['tc'][:m+1,q])  
                  
                   if len(mesh['eb']) > 0:
                    np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-np.einsum('mc,mc->c',Xm[:,mesh['eb']],GG[:m+1,q]))
                   
                   Jp += np.einsum('mc,mj->cj',Xm,sigma[:m+1,q,0:argv['dim']])*1e-18
                   break

                 DeltaTp += X*mat['tc'][m,q]

                 if len(mesh['eb']) > 0:
                  np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-X[mesh['eb']]*GG[m,q])

                 Jp += np.einsum('c,j->cj',X,sigma[m,q,0:argv['dim']])*1e-18
                
 
           for m in range(argv['n_serial'])[idx+1:]:

               #----------------------------------------------------
               B = DeltaT +  mfp[m]*(P[n] +get_boundary(RHS[n],mesh['eb'],n_elems))
               A = get_m(Master,np.concatenate((mfp[m]*Gm[n],mfp[m]*D[n]+np.ones(n_elems)))[conversion])
               X = solve(argv,A,B,(q,m))
               kappap[m,q] = np.dot(mesh['kappa_mask'],X)
               #-------------------------------------------------------

               error_bal = abs(kappap[m,q] - kappap_b[m,q])/abs(kappap[m,q])
               if error_bal < argv['multiscale_error_ballistic'] and \
                   abs(kappap[m-1,q] - kappap_b[m-1,q])/abs(kappap[m-1,q]) < argv['multiscale_error_ballistic'] and  m > int(len(mat['mfp_sampled'])/2):

                   kappap[m:,q] = kappap_b[m:,q]
                   bal += argv['n_serial']-m
                   DeltaTp += X_bal*np.sum(mat['tc'][m:,q])

                   if len(mesh['eb']) > 0:
                    np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-np.einsum('c,mc->c',X_bal[mesh['eb']],GG[m:,q]))

                   Jp += np.einsum('c,j->cj',X_bal,np.sum(sigma[m:,q,0:argv['dim']],axis=0))*1e-18
                   break

               DeltaTp += X*mat['tc'][m,q]
               
               if len(mesh['eb'])>0:
                np.add.at(TBp,np.arange(mesh['eb'].shape[0]),-X[mesh['eb']]*GG[m,q])

               Jp += np.einsum('c,j->cj',X,sigma[m,q,0:argv['dim']])*1e-18
          

        Mp[0] = diffusive
        Mp[1] = bal

        #Centering--
        #DeltaT = DeltaT - (max(DeltaT)+min(DeltaT))/2.0

        DeltaT_old = DeltaT.copy()

        comm.Barrier()

        comm.Allreduce([DeltaTp,MPI.DOUBLE],[DeltaT,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Jp,MPI.DOUBLE],[J,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappap,MPI.DOUBLE],[kappa,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappap_b,MPI.DOUBLE],[kappab,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappap_0,MPI.DOUBLE],[kappa0,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappap_f,MPI.DOUBLE],[kappaf,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([Mp,MPI.DOUBLE],[MM,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([transitionp,MPI.DOUBLE],[transition,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([TBp,MPI.DOUBLE],[TB,MPI.DOUBLE],op=MPI.SUM)
        kappa_totp = np.array([np.einsum('mq,mq->',sigma[:,argv['rr'],int(mesh['meta'][-1])],kappa[:,argv['rr']])])
        kappaf_totp = np.array([np.einsum('mq,mq->',sigma[:,argv['rr'],int(mesh['meta'][-1])],kappaf[:,argv['rr']])])
        comm.Allreduce([kappa_totp,MPI.DOUBLE],[kappa_tot,MPI.DOUBLE],op=MPI.SUM)
        comm.Allreduce([kappaf_totp,MPI.DOUBLE],[kappaf_tot,MPI.DOUBLE],op=MPI.SUM)

        if argv['multiscale'] and comm.rank == 0: print_multiscale(argv,MM) 

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
    output =  {'kappa':kappa_vec,'temperature':DeltaT,'flux':J,'kappa_mode':kappa,'kappa_mode_f':kappaf,\
                'kappa_mode_b':kappab-np.dot(mesh['kappa_mask'],DeltaT_old)}



    return output



          
     








