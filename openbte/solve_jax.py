from __future__ import absolute_import
import numpy as np
from scipy.sparse.linalg import splu
from termcolor import colored, cprint 
from .utils import *
from .fourier import *
import deepdish as dd
from mpi4py import MPI
import scipy.sparse as sp
import time
import sys
import scipy
from jax import random
from gmres_google import *
from gmres_mit import *


comm = MPI.COMM_WORLD


      
def get_m(Master,data):
    
    Master.data = data
    return Master

def get_boundary(RHS,eb,n_elems):

    RHS_tmp = np.zeros(n_elems) 
    for c,i in enumerate(eb): RHS_tmp[i] += RHS[c]

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


@functools.partial(jax.jit, static_argnums=(3,4,5))
def get_boundary_jit(RHS,P,DeltaT_old,mfp,eb,n_elems,m,n):

 RHS_tmp = jnp.zeros(n_elems)
 for c,i in enumerate(eb) : RHS_tmp = jax.ops.index_add(RHS_tmp,jax.ops.index[i],RHS[m,n,c])

 b = DeltaT_old + mfp[m]*(P[n] +  RHS_tmp)

 return b





@functools.partial(jax.jit, static_argnums=(4,5,6,7))
def update_values(X,TB,DeltaT,kappa,tc,kappa_mask,eb,GG,m,q) :

    kappa += jnp.dot(kappa_mask,X)
    DeltaT += X*tc[m,q]
    for c,i in enumerate(eb): TB = jax.ops.index_add(TB,jax.ops.index[m,c],X[i]*GG[m,q,c])

    return DeltaT,TB,kappa



def solve_jax(**argv):

    mesh    = argv['mesh']
    mat     = argv['mat']
    fourier = argv['fourier']
    n_elems = len(mesh['elems'])
    mfp = mat['mfp_sampled']

    if comm.rank == 0 and argv['verbose']:
       print(flush=True)
       print('      Iter    Thermal Conductivity [W/m/K]      Error ''',flush=True)
       print(colored(' -----------------------------------------------------------','green'),flush=True)

    if len(mesh['db']) > 0:
      if comm.rank == 0:  
       Gb   = np.einsum('mqj,jn->mqn',mat['sigma'],mesh['db'],optimize=True)
       Gbp2 = Gb.clip(min=0);
       with np.errstate(divide='ignore', invalid='ignore'):
          tot = 1/Gb.clip(max=0).sum(axis=1); tot[np.isinf(tot)] = 0
       data = {'GG': np.einsum('mqs,ms->mqs',Gbp2,tot)}
       del tot, Gbp2
      else: data = None
      GG = create_shared_memory_dict(data)['GG']
    
    #Bulk properties---
    G = np.einsum('qj,jn->qn',mat['VMFP'][argv['rr']],mesh['k'],optimize=True)
    Gp = G.clip(min=0); Gm = G.clip(max=0)
    D = np.zeros((len(argv['rr']),len(mesh['elems'])))
    for n,i in enumerate(mesh['i']):  D[:,i] += Gp[:,n]
    DeltaT_old = fourier['temperature_fourier'].copy()

    #---boundary----------------------
    TB_old = np.zeros((argv['n_serial'],len(mesh['eb'])))   
    if len(mesh['db']) > 0: #boundary
     tmp = np.einsum('rj,jn->rn',mat['VMFP'][argv['rr']],mesh['db'],optimize=True)  
     Gbp2 = tmp.clip(min=0); Gbm2 = tmp.clip(max=0);
     for n,i in enumerate(mesh['eb']):
         D[:, i]  += Gbp2[:,n]
         TB_old[:,n]  = DeltaT_old[i]  
    #Periodic---
    P = np.zeros((len(argv['rr']),n_elems))
    for ss,v in mesh['pp']: P[:,mesh['i'][int(ss)]] -= Gm[:,int(ss)].copy()*v
    del G,Gp
    #----------------------------------------------------

    lu =  {}
    kappa_vec = [fourier['meta'][0]]
    kappa_old = kappa_vec[-1]
    error = 1
    kk = 0
    kappa_tot = np.zeros(1)


    Master = sp.csc_matrix((np.arange(len(mesh['im']))+1,(mesh['im'],mesh['jm'])),shape=(n_elems,n_elems),dtype=np.float64)
    conversion = np.asarray(Master.data-1,np.int) 


    #put material data to device

    kappa_mask = jnp.asarray(mesh['kappa_mask']*argv['kappa_factor']*1e-18)
    tc = jnp.asarray(mat['tc'])#.reshape(argv['n_serial']*argv['n_parallel'])
    GG = jnp.asarray(GG)
    P = jnp.asarray(P)
    mfp = jnp.asarray(mfp)
    eb = jnp.asarray(mesh['eb'])
    DeltaT_old = jnp.asarray(DeltaT_old)
    TB_old = jnp.asarray(TB_old)

    cache = {}
    kappa = 0
    while kk < argv['max_bte_iter'] and error > argv['max_bte_error']:
       
        #RHS = -np.einsum('mc,nc->mnc',TB_old,Gbm2) if len(mesh['db']) > 0 else  np.zeros((n_serial,n_parallel,0))   
        #TB = np.zeros_like(TB_old)
        #DeltaT = np.zeros_like(DeltaT_old)
        RHS = -jnp.einsum('mc,nc->mnc',TB_old,Gbm2) if len(mesh['db']) > 0 else  np.zeros((n_serial,n_parallel,0))   
        DeltaT = jnp.zeros_like(DeltaT_old)
        TB = jnp.zeros_like(TB_old)

        for n,q in enumerate(argv['rr']):
           
           fourier = False
           for m in range(argv['n_serial']):

                #a= time.time()
                #X = (lu[(m,n)] if (m,n) in lu.keys() else \
                #     lu.setdefault((m,n),sp.linalg.splu(get_m(Master,np.concatenate((mfp[m]*Gm[n],\
                #     mfp[m]*D[n]+np.ones(n_elems)))[conversion])))).solve(DeltaT_old +  mfp[m]*(P[n] +\
                #     get_boundary(RHS[m,n],mesh['eb'],n_elems))) 
                #kappa += np.dot(mesh['kappa_mask'],X)
                #DeltaT += X*mat['tc'][m,q]
                #for c,i in enumerate(mesh['eb']): 
                #    TB[m,c] -= X[i]*GG[m,q,c]
                #print(time.time()-a)    


                #a= time.time()
                #print(time.time()-a)
                #print(time.time()-a)     
                #X = sp.linalg.splu(A).solve(b)
                #X = scipy.sparse.linalg.spsolve(A,b)
                #X = scipy.sparse.linalg.gmres(A,b)[0]
                #X = gmres_mit(A.toarray(),b,np.zeros_like(b))
                #X = gmres(functools.partial(jnp.dot, A.todense()), b, n=20)
                #X = scipy.sparse.linalg.gmres(np.array(A), np.array(b), restart=20, maxiter=1)[0]
 
                       
                a= time.time()

                A = get_m(Master,np.concatenate((mfp[m]*Gm[n],mfp[m]*D[n]+np.ones(n_elems)))[conversion]) #need to be jitted
                
                def sparse_dot(i,j,k,b):
                  
                    
                #Dot product
                for i,j,k in zip(im,jm,km): A[i] += k*b[j]

                quit()
                b = get_boundary_jit(RHS,P,DeltaT_old,mfp,eb,n_elems,m,n) #jitted
                X = explicit_gmres(A.todense(), b, 10).block_until_ready() #jitted
                DeltaT,TB,kappa = update_values(X,TB,DeltaT,kappa,tc,kappa_mask,eb,GG,m,q) #jitted
                print(time.time()-a)    

                quit()

                #J += np.einsum('c,j->cj',X,mat['sigma'][m,q,0:argv['dim']])*1e-18
                #Sup += kappa[m,q]*mat['suppression'][m,q,:]*argv['kappa_factor']*1e-9

        #DeltaT_old = DeltaT.copy()
        #TB_old = TB.copy()
        #kappa_tot = np.array([np.einsum('mq,mq->',mat['sigma'][:,argv['rr'],0],kappa[:,argv['rr']])])*argv['kappa_factor']*1e-18
        kk +=1
        error = abs(kappa_old-kappa_tot[0])/abs(kappa_tot[0])
        kappa_old = kappa_tot[0]
        kappa_vec.append(kappa_tot[0])
        if argv['verbose'] and comm.rank == 0:   
         print('{0:8d} {1:24.4E} {2:22.4E}'.format(kk,kappa_vec[-1],error),flush=True)

    if argv['verbose'] and comm.rank == 0:
      print(colored(' -----------------------------------------------------------','green'),flush=True)


          


     








